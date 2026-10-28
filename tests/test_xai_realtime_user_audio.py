#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for user-audio and interruption handling in GrokRealtimeLLMService."""

import asyncio
import base64
from types import SimpleNamespace
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
    service.create_task = lambda coroutine, name: asyncio.create_task(coroutine, name=name)
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
    recovery = service._audio_commit_fallback
    await service._handle_evt_audio_committed(None)
    await asyncio.wait_for(recovery, 1)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["recommit", "restore"])
async def test_late_acknowledgements_cannot_cancel_replay_or_mix_resumed_speech(monkeypatch, phase):
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.xai.realtime import llm

    monkeypatch.setattr(llm, "AUDIO_COMMIT_ACK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(llm, "REPLAY_APPEND_BYTES", 4)
    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    turn, following, resumed = b"abcdefgh", b"ij", b"kl"
    reached, release = asyncio.Event(), asyncio.Event()
    replaying = False
    recommitting = False

    async def send(event):
        nonlocal replaying, recommitting
        await recorder(event)
        current = None
        if isinstance(event, events.InputAudioBufferClearEvent):
            replaying = True
            service._input_cleared.set()
            current = "clear"
        elif replaying and isinstance(event, events.InputAudioBufferCommitEvent):
            recommitting = True
            current = "recommit"
        elif replaying and isinstance(event, events.InputAudioBufferAppendEvent):
            current = "restore" if recommitting else "append"
        if current == phase and not reached.is_set():
            reached.set()
            await release.wait()

    service.send_client_event = send
    await service._send_user_audio(_audio_frame(turn))
    await service._handle_user_stopped_speaking(None)
    await service._send_user_audio(_audio_frame(following))
    recovery = service._audio_commit_fallback
    await asyncio.wait_for(reached.wait(), 1)
    await service._handle_evt_audio_committed(None)
    await service._handle_evt_audio_committed(None)
    await service._send_user_audio(_audio_frame(resumed))
    assert not recovery.cancelled() and not recovery.done()
    assert "ResponseCreateEvent" not in recorder.kinds()
    release.set()
    await asyncio.wait_for(recovery, 1)
    kinds = recorder.kinds()
    clear = kinds.index("InputAudioBufferClearEvent")
    commit = kinds.index("InputAudioBufferCommitEvent", clear)
    assert b"".join(base64.b64decode(e.audio) for e in recorder.events[clear + 1 : commit]) == turn
    after = [
        e
        for e in recorder.events[commit + 1 :]
        if isinstance(e, events.InputAudioBufferAppendEvent)
    ]
    assert b"".join(base64.b64decode(e.audio) for e in after) == following + resumed
    assert service._turn_audio == following + resumed
    assert kinds.count("InputAudioBufferCommitEvent") == 2
    assert kinds.count("ResponseCreateEvent") == 1
    await service._handle_evt_audio_committed(None)
    assert recorder.kinds().count("ResponseCreateEvent") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["clear", "append"])
async def test_original_acknowledged_before_recommit_stops_the_replay(monkeypatch, phase):
    """A slow original acknowledgement must not commit the same turn twice."""
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.xai.realtime import llm

    monkeypatch.setattr(llm, "AUDIO_COMMIT_ACK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(llm, "REPLAY_APPEND_BYTES", 4)
    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    turn, following, resumed = b"abcdefgh", b"ij", b"kl"
    reached, release = asyncio.Event(), asyncio.Event()
    replaying = False

    async def send(event):
        nonlocal replaying
        await recorder(event)
        current = None
        if isinstance(event, events.InputAudioBufferClearEvent):
            service._input_cleared.set()
            current = "clear" if not replaying else None
            replaying = True
        elif replaying and isinstance(event, events.InputAudioBufferAppendEvent):
            current = "append"
        if current == phase and not reached.is_set():
            reached.set()
            await release.wait()

    service.send_client_event = send
    await service._send_user_audio(_audio_frame(turn))
    await service._handle_user_stopped_speaking(None)
    await service._send_user_audio(_audio_frame(following))
    recovery = service._audio_commit_fallback
    await asyncio.wait_for(reached.wait(), 1)
    await service._handle_evt_audio_committed(None)
    await service._send_user_audio(_audio_frame(resumed))
    release.set()
    await asyncio.wait_for(recovery, 1)
    kinds = recorder.kinds()
    assert kinds.count("InputAudioBufferCommitEvent") == 1
    last_clear = len(kinds) - 1 - kinds[::-1].index("InputAudioBufferClearEvent")
    if phase == "append":
        # The partial replay is cleared rather than left for the next turn.
        assert kinds.count("InputAudioBufferClearEvent") == 2
    else:
        # Acknowledged before any replay bytes were sent: nothing to clear again.
        assert kinds.count("InputAudioBufferClearEvent") == 1
    restored = [
        e
        for e in recorder.events[last_clear + 1 :]
        if isinstance(e, events.InputAudioBufferAppendEvent)
    ]
    assert b"".join(base64.b64decode(e.audio) for e in restored) == following + resumed
    assert service._turn_audio == following + resumed
    assert kinds.count("ResponseCreateEvent") == 1
    assert not service._audio_commit_in_progress


@pytest.mark.asyncio
async def test_commit_behind_a_cancelled_response_is_not_replayed_while_it_finishes(monkeypatch):
    """The API acknowledges a commit only after a cancelled response is done."""
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.xai.realtime import llm

    monkeypatch.setattr(llm, "AUDIO_COMMIT_ACK_TIMEOUT_SECONDS", 0.05)
    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    service._response_active = True
    service._current_response_id = "speaking"
    await service._handle_interruption()
    await service._send_user_audio(_audio_frame(b"ab"))
    await service._handle_user_stopped_speaking(None)
    recovery = service._audio_commit_fallback
    # Well past the acknowledgement bound, the cancelled response still runs.
    await asyncio.sleep(0.2)
    assert "InputAudioBufferClearEvent" not in recorder.kinds()
    cancelled = SimpleNamespace(
        usage=None, response=SimpleNamespace(usage=None, status="cancelled", output=[])
    )
    await service._handle_evt_response_done(cancelled)
    await service._handle_evt_audio_committed(None)
    await asyncio.sleep(0.1)
    assert recovery.cancelled() or recovery.done()
    kinds = recorder.kinds()
    assert "InputAudioBufferClearEvent" not in kinds
    assert kinds.count("InputAudioBufferCommitEvent") == 1


@pytest.mark.asyncio
async def test_a_cancel_that_never_finishes_still_allows_recovery(monkeypatch):
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.xai.realtime import llm

    monkeypatch.setattr(llm, "AUDIO_COMMIT_ACK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(llm, "CANCEL_SETTLE_TIMEOUT_SECONDS", 0.05)
    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()

    async def send(event):
        await recorder(event)
        if isinstance(event, events.InputAudioBufferClearEvent):
            service._input_cleared.set()

    service.send_client_event = send
    service._response_active = True
    await service._handle_interruption()
    await service._send_user_audio(_audio_frame(b"ab"))
    await service._handle_user_stopped_speaking(None)
    await asyncio.sleep(0.2)
    kinds = recorder.kinds()
    assert "InputAudioBufferClearEvent" in kinds
    assert kinds.count("InputAudioBufferCommitEvent") == 2


@pytest.mark.asyncio
async def test_next_completed_turn_waits_for_recovery_and_is_committed_separately(monkeypatch):
    from unittest.mock import AsyncMock

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.xai.realtime import llm

    monkeypatch.setattr(llm, "AUDIO_COMMIT_ACK_TIMEOUT_SECONDS", 0.01)
    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    service._llm_needs_conversation_setup = False
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    reached, release = asyncio.Event(), asyncio.Event()

    async def send(event):
        await recorder(event)
        if isinstance(event, events.InputAudioBufferClearEvent):
            reached.set()
            await release.wait()
            service._input_cleared.set()

    service.send_client_event = send
    await service._send_user_audio(_audio_frame(b"ab"))
    await service._handle_user_stopped_speaking(None)
    recovery = service._audio_commit_fallback
    await asyncio.wait_for(reached.wait(), 1)
    await service._handle_user_started_speaking(None)
    await service._send_user_audio(_audio_frame(b"cd"))
    await service._handle_user_stopped_speaking(None)
    assert service._pending_commit_audio == b"ab"
    release.set()
    await asyncio.wait_for(recovery, 1)
    assert service._pending_commit_audio == b"cd"
    assert recorder.kinds().count("InputAudioBufferCommitEvent") == 3
    await service._handle_evt_audio_committed(None)
    assert recorder.kinds().count("ResponseCreateEvent") == 1


@pytest.mark.asyncio
async def test_disconnect_joins_replay_before_clearing_audio_state(monkeypatch):
    from unittest.mock import AsyncMock

    from pipecat.services.xai.realtime import llm

    monkeypatch.setattr(llm, "AUDIO_COMMIT_ACK_TIMEOUT_SECONDS", 0.01)
    service, recorder = _make_service(server_vad=False)
    service._api_session_ready = True
    reached = asyncio.Event()

    async def send(event):
        await recorder(event)
        if isinstance(event, events.InputAudioBufferClearEvent):
            reached.set()
            await asyncio.Event().wait()

    service.send_client_event = send
    service.push_frame = AsyncMock()
    await service._send_user_audio(_audio_frame(b"ab"))
    await service._handle_user_stopped_speaking(None)
    recovery = service._audio_commit_fallback
    await asyncio.wait_for(reached.wait(), 1)
    await service._send_user_audio(_audio_frame(b"cd"))
    await service._disconnect()
    assert recovery.done() and recovery.cancelled()
    assert not service._audio_commit_in_progress and not service._audio_recovering
    assert not service._turn_audio and not service._recovery_audio
    assert recorder.kinds().count("InputAudioBufferCommitEvent") == 1
