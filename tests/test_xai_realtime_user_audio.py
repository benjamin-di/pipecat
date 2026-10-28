#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for user-audio and interruption handling in GrokRealtimeLLMService."""

import asyncio
import base64
from typing import Any

import pytest

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.xai.realtime import events
from pipecat.services.xai.realtime.events import SessionProperties, TurnDetection
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService


class _EventRecorder:
    def __init__(self):
        self.events: list[Any] = []

    async def __call__(self, event):
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [type(e).__name__ for e in self.events]


def _make_service(*, server_vad: bool) -> tuple[GrokRealtimeLLMService, _EventRecorder]:
    turn_detection = TurnDetection(type="server_vad") if server_vad else None
    service = GrokRealtimeLLMService(
        api_key="test-key",
        settings=GrokRealtimeLLMService.Settings(
            session_properties=SessionProperties(turn_detection=turn_detection),
        ),
    )
    recorder = _EventRecorder()
    service.send_client_event = recorder  # type: ignore[method-assign]

    async def _noop(*args, **kwargs):
        pass

    service.stop_all_metrics = _noop  # type: ignore[method-assign]
    return service, recorder


def _audio_frame(data: bytes = b"\xaa\xbb") -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=data, sample_rate=24000, num_channels=1)


def test_default_model_is_grok_voice_latest():
    service = GrokRealtimeLLMService(api_key="test-key")
    assert service._settings.model == "grok-voice-latest"


@pytest.mark.asyncio
@pytest.mark.parametrize("server_vad", [False, True])
async def test_session_update_serializes_explicit_manual_mode(server_vad):
    from unittest.mock import AsyncMock

    props = SessionProperties(
        turn_detection=TurnDetection(type="server_vad") if server_vad else None
    )
    service = GrokRealtimeLLMService(api_key="test-key")
    service._ws_send = AsyncMock()
    await service.send_client_event(events.SessionUpdateEvent(session=props))
    actual = service._ws_send.await_args.args[0]["session"]["turn_detection"]
    assert actual == ({"type": "server_vad"} if server_vad else None)


@pytest.mark.asyncio
async def test_user_audio_dropped_until_session_ready():
    from unittest.mock import patch

    service, recorder = _make_service(server_vad=True)
    assert service._api_session_ready is False

    with patch("pipecat.services.xai.realtime.llm.logger.debug") as mock_debug:
        await service._send_user_audio(_audio_frame())
        await service._send_user_audio(_audio_frame())

    assert recorder.kinds() == []
    assert service._logged_audio_drop_before_session_ready is True
    drop_calls = [c for c in mock_debug.call_args_list if "Dropping user audio" in str(c)]
    assert len(drop_calls) == 1


@pytest.mark.asyncio
async def test_user_audio_flows_after_session_ready_without_conversation_setup():
    """Audio-only pipelines never call _create_response; audio must still flow."""
    service, recorder = _make_service(server_vad=True)
    service._api_session_ready = True
    assert service._llm_needs_conversation_setup is True

    await service._send_user_audio(_audio_frame(b"\x11\x22"))

    assert recorder.kinds() == ["InputAudioBufferAppendEvent"]
    assert recorder.events[0].audio == base64.b64encode(b"\x11\x22").decode()


@pytest.mark.asyncio
async def test_server_vad_interruption_cancels_without_clearing_input():
    service, recorder = _make_service(server_vad=True)
    service._api_session_ready = True

    await service._send_user_audio(_audio_frame())
    await service._handle_interruption()

    assert recorder.kinds() == [
        "InputAudioBufferAppendEvent",
        "ResponseCancelEvent",
    ]
    assert "InputAudioBufferClearEvent" not in recorder.kinds()


@pytest.mark.asyncio
async def test_manual_turn_interruption_preserves_the_interrupting_user_audio():
    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    await service._send_user_audio(_audio_frame())
    await service._handle_interruption()

    assert recorder.kinds() == [
        "InputAudioBufferAppendEvent",
        "ResponseCancelEvent",
    ]


@pytest.mark.asyncio
async def test_tool_reply_waits_for_manual_user_turn_commit():
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext

    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    await service._handle_user_started_speaking(None)
    service._results_awaiting_response = True
    await service._create_response()
    assert recorder.kinds() == []
    assert service._response_create_owed
    await service._handle_user_stopped_speaking(None)
    # The API can start a response before it processes the commit; the reply
    # would then omit the user's turn. Wait for the acknowledgement.
    assert recorder.kinds() == ["InputAudioBufferCommitEvent"]
    await service._handle_evt_audio_committed(None)
    assert recorder.kinds() == ["InputAudioBufferCommitEvent", "ResponseCreateEvent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("resumes_speaking", [False, True])
async def test_tool_reply_cannot_overtake_an_in_progress_audio_commit(resumes_speaking):
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext

    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    committing, release = asyncio.Event(), asyncio.Event()

    async def send(event):
        if isinstance(event, events.InputAudioBufferCommitEvent):
            committing.set()
            await release.wait()
        await recorder(event)

    service.send_client_event = send
    await service._handle_user_started_speaking(None)
    stopped = asyncio.create_task(service._handle_user_stopped_speaking(None))
    try:
        await asyncio.wait_for(committing.wait(), 1)
        service._results_awaiting_response = True
        await service._create_response()
        before_commit = recorder.kinds()
        if resumes_speaking:
            await service._handle_user_started_speaking(None)
    finally:
        release.set()
        await stopped

    assert before_commit == []
    assert recorder.kinds() == ["InputAudioBufferCommitEvent"]
    await service._handle_evt_audio_committed(None)
    assert recorder.kinds() == ["InputAudioBufferCommitEvent"] + (
        [] if resumes_speaking else ["ResponseCreateEvent"]
    )
    if resumes_speaking:
        assert service._user_speaking
        await service._handle_user_stopped_speaking(None)
        await service._handle_evt_audio_committed(None)
        assert recorder.kinds() == [
            "InputAudioBufferCommitEvent",
            "InputAudioBufferCommitEvent",
            "ResponseCreateEvent",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledged_first", [True, False])
async def test_manual_response_waits_for_previous_response_to_finish(acknowledged_first):
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext

    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    service._response_active = True
    await service._handle_user_stopped_speaking(None)
    assert recorder.kinds() == ["InputAudioBufferCommitEvent"]
    done = events.ResponseDone(
        type="response.done",
        event_id="event-done",
        response=events.Response(
            id="response-a", object="realtime.response", status="completed", output=[]
        ),
    )
    # Either order of acknowledgement and response completion yields one reply.
    if acknowledged_first:
        await service._handle_evt_audio_committed(None)
        assert recorder.kinds() == ["InputAudioBufferCommitEvent"]
        await service._handle_evt_response_done(done)
    else:
        await service._handle_evt_response_done(done)
        assert recorder.kinds() == ["InputAudioBufferCommitEvent"]
        await service._handle_evt_audio_committed(None)
    assert recorder.kinds() == ["InputAudioBufferCommitEvent", "ResponseCreateEvent"]


@pytest.mark.asyncio
async def test_unacknowledged_manual_commit_releases_the_response_after_its_bound(monkeypatch):
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.xai.realtime import llm

    monkeypatch.setattr(llm, "AUDIO_COMMIT_ACK_TIMEOUT_SECONDS", 0.05)
    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    await service._handle_user_stopped_speaking(None)
    assert recorder.kinds() == ["InputAudioBufferCommitEvent"]
    await asyncio.sleep(0.2)
    assert recorder.kinds() == ["InputAudioBufferCommitEvent", "ResponseCreateEvent"]
    # A late acknowledgement after the fallback starts nothing further.
    await service._handle_evt_audio_committed(None)
    assert recorder.kinds() == ["InputAudioBufferCommitEvent", "ResponseCreateEvent"]


@pytest.mark.asyncio
async def test_ignored_manual_commit_is_replayed_after_clearing_the_buffer(monkeypatch):
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.xai.realtime import llm

    monkeypatch.setattr(llm, "AUDIO_COMMIT_ACK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(llm, "REPLAY_APPEND_BYTES", 4)
    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()

    async def send(event):
        await recorder(event)
        # The API acknowledges clearing, then the replayed commit.
        if isinstance(event, events.InputAudioBufferClearEvent):
            service._input_cleared.set()

    service.send_client_event = send
    turn = b"\x01\x02\x03\x04\x05\x06"
    await service._send_user_audio(_audio_frame(turn))
    await service._handle_user_stopped_speaking(None)
    # Speech after the commit belongs to the next turn and must survive the clear.
    await service._send_user_audio(_audio_frame(b"\x07\x08"))
    await asyncio.sleep(0.08)
    kinds = recorder.kinds()
    assert kinds == [
        "InputAudioBufferAppendEvent",
        "InputAudioBufferCommitEvent",
        "InputAudioBufferAppendEvent",
        "InputAudioBufferClearEvent",
        "InputAudioBufferAppendEvent",
        "InputAudioBufferAppendEvent",
        "InputAudioBufferCommitEvent",
        "InputAudioBufferAppendEvent",
    ]
    replayed = b"".join(base64.b64decode(e.audio) for e in recorder.events[4:6])
    assert replayed == turn
    assert base64.b64decode(recorder.events[7].audio) == b"\x07\x08"
    # No reply is requested until the replayed commit is acknowledged.
    assert "ResponseCreateEvent" not in kinds
    await service._handle_evt_audio_committed(None)
    assert recorder.kinds()[-1] == "ResponseCreateEvent"
    assert recorder.kinds().count("ResponseCreateEvent") == 1
    await asyncio.sleep(0.1)
    assert recorder.kinds().count("ResponseCreateEvent") == 1
    assert service._turn_audio == bytearray(b"\x07\x08")


@pytest.mark.asyncio
async def test_unsolicited_commit_acknowledgement_starts_no_response():
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext

    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    await service._handle_evt_audio_committed(None)
    assert recorder.kinds() == []
    assert not service._audio_commit_in_progress


@pytest.mark.asyncio
async def test_late_cancelled_audio_and_transcript_cannot_restart_playback():
    from unittest.mock import AsyncMock

    from pipecat.frames.frames import TTSAudioRawFrame

    service, recorder = _make_service(server_vad=True)
    service.push_frame = AsyncMock()
    service._settings.session_properties.audio = events.AudioConfiguration(
        output=events.AudioOutput(format=events.PCMAudioFormat(rate=24000))
    )
    payload = dict(event_id="event-a", item_id="audio-a", output_index=0, content_index=0)

    def audio(response):
        return events.ResponseAudioDelta(
            type="response.output_audio.delta",
            response_id=response,
            delta=base64.b64encode(bytes(960)).decode(),
            **payload,
        )

    await service._handle_evt_audio_delta(audio("response-a"))
    await service._handle_interruption()
    before = service.push_frame.await_count
    await service._handle_evt_audio_delta(audio("response-a"))
    await service._handle_evt_audio_transcript_delta(
        events.ResponseAudioTranscriptDelta(
            type="response.output_audio_transcript.delta",
            response_id="response-a",
            delta="Stale response.",
            **payload,
        )
    )
    assert service.push_frame.await_count == before
    await service._handle_evt_audio_delta(audio("response-b"))
    assert (
        len(
            [
                c
                for c in service.push_frame.await_args_list
                if isinstance(c.args[0], TTSAudioRawFrame)
            ]
        )
        == 2
    )


@pytest.mark.asyncio
async def test_interruption_truncates_in_flight_audio_on_the_wire():
    import time

    from pipecat.services.xai.realtime.events import (
        AudioConfiguration,
        AudioOutput,
        PCMAudioFormat,
    )
    from pipecat.services.xai.realtime.llm import CurrentAudioResponse

    service, recorder = _make_service(server_vad=True)
    service._settings.session_properties.audio = AudioConfiguration(
        output=AudioOutput(format=PCMAudioFormat(rate=24000))
    )
    service._current_audio_response = CurrentAudioResponse(
        item_id="item-audio",
        content_index=0,
        start_time_ms=int(time.time() * 1000) - 500,
        total_size=48000,  # 1s at 24kHz mono 16-bit
    )

    await service._handle_interruption()

    assert "ResponseCancelEvent" in recorder.kinds()
    assert "ConversationItemTruncateEvent" in recorder.kinds()
    truncate = next(
        e for e in recorder.events if isinstance(e, events.ConversationItemTruncateEvent)
    )
    assert truncate.item_id == "item-audio"
    assert truncate.content_index == 0
    assert 0 < truncate.audio_end_ms <= 1000
    assert service._current_audio_response is None


@pytest.mark.asyncio
async def test_delete_conversation_item_sends_client_event():
    service, recorder = _make_service(server_vad=True)
    await service.delete_conversation_item("item-1")
    assert recorder.kinds() == ["ConversationItemDeleteEvent"]
    assert recorder.events[0].item_id == "item-1"


@pytest.mark.asyncio
async def test_force_message_sends_force_message_item():
    service, recorder = _make_service(server_vad=True)
    await service.force_message("This call is being recorded.")
    assert recorder.kinds() == ["ConversationItemCreateEvent"]
    item = recorder.events[0].item
    assert item.type == "force_message"
    assert item.content[0].text == "This call is being recorded."
