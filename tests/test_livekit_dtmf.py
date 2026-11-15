# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Native LiveKit DTMF sequences and observable send failures."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    OutputAudioRawFrame,
    OutputDTMFFrame,
    OutputDTMFUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.livekit.transport import (
    LiveKitOutputTransport,
    LiveKitParams,
    LiveKitTransportClient,
)
from pipecat.utils.asyncio.task_manager import TaskManager


class TestLiveKitDTMF(unittest.IsolatedAsyncioTestCase):
    async def test_every_digit_is_sent_in_order(self):
        for frame_type in (OutputDTMFFrame, OutputDTMFUrgentFrame):
            sent = []

            async def send(digit, sent=sent):
                sent.append(digit)

            transport = SimpleNamespace(_client=SimpleNamespace(send_dtmf=send))
            frame = frame_type(buttons=[KeypadEntry.ONE, KeypadEntry.TWO, KeypadEntry.POUND])
            await LiveKitOutputTransport._write_dtmf_native(transport, frame)
            self.assertEqual(sent, ["1", "2", "#"])

    async def test_failure_stops_remaining_digits_and_is_observable(self):
        for frame_type in (OutputDTMFFrame, OutputDTMFUrgentFrame):
            sent = []
            failure = OSError("gateway unavailable")

            async def send(digit, sent=sent, failure=failure):
                sent.append(digit)
                if digit == "2":
                    raise failure

            transport = SimpleNamespace(
                _client=SimpleNamespace(send_dtmf=send), push_error=AsyncMock()
            )
            frame = frame_type(buttons=[KeypadEntry.ONE, KeypadEntry.TWO, KeypadEntry.POUND])
            await LiveKitOutputTransport._write_dtmf_native(transport, frame)
            self.assertEqual(sent, ["1", "2"])
            transport.push_error.assert_awaited_once()
            self.assertIs(transport.push_error.await_args.kwargs["exception"], failure)

    async def test_cancelled_send_is_not_reported_as_a_publication_failure(self):
        transport = SimpleNamespace(
            _client=SimpleNamespace(send_dtmf=AsyncMock(side_effect=asyncio.CancelledError)),
            push_error=AsyncMock(),
        )
        frame = OutputDTMFFrame(buttons=[KeypadEntry.ONE, KeypadEntry.TWO])
        with self.assertRaises(asyncio.CancelledError):
            await LiveKitOutputTransport._write_dtmf_native(transport, frame)
        transport._client.send_dtmf.assert_awaited_once_with("1")
        transport.push_error.assert_not_awaited()

    async def test_failed_queued_dtmf_does_not_stop_following_audio(self):
        client = SimpleNamespace(
            setup=AsyncMock(),
            connect=AsyncMock(),
            disconnect=AsyncMock(),
            send_dtmf=AsyncMock(side_effect=[None, OSError("gateway unavailable")]),
        )
        output = LiveKitOutputTransport(
            SimpleNamespace(cleanup=AsyncMock()), client, LiveKitParams(audio_out_enabled=True)
        )
        output.push_frame = AsyncMock()
        output.push_error = AsyncMock(wraps=output.push_error)
        written = asyncio.Event()

        async def write(frame):
            written.set()
            return True

        output.write_audio_frame = AsyncMock(side_effect=write)
        await output.setup(
            FrameProcessorSetup(
                clock=SystemClock(),
                task_manager=TaskManager(),
                pipeline_worker=SimpleNamespace(app_resources=None),
            )
        )
        try:
            await output.process_frame(
                StartFrame(audio_out_sample_rate=16000), FrameDirection.DOWNSTREAM
            )
            sender = output._media_senders[None]
            await output.process_frame(
                OutputDTMFFrame(buttons=[KeypadEntry.ONE, KeypadEntry.TWO, KeypadEntry.POUND]),
                FrameDirection.DOWNSTREAM,
            )
            audio = OutputAudioRawFrame(b"\x01" * sender.audio_chunk_size, sender.sample_rate, 1)
            await output.process_frame(audio, FrameDirection.DOWNSTREAM)
            await asyncio.wait_for(written.wait(), 1)
            self.assertFalse(sender._audio_task.done())
            self.assertEqual(client.send_dtmf.await_count, 2)
            output.push_error.assert_awaited_once()
        finally:
            await output.cancel(CancelFrame())
            await output.cleanup()

    async def test_client_publishes_correct_code(self):
        publish = AsyncMock()
        client = SimpleNamespace(
            _connected=True,
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish)),
        )
        await LiveKitTransportClient.send_dtmf(client, "#")
        publish.assert_awaited_once_with(code=11, digit="#")

    async def test_client_does_not_swallow_gateway_failure(self):
        client = SimpleNamespace(
            _connected=True,
            room=SimpleNamespace(
                local_participant=SimpleNamespace(publish_dtmf=AsyncMock(side_effect=OSError))
            ),
        )
        with self.assertRaises(OSError):
            await LiveKitTransportClient.send_dtmf(client, "1")

    async def test_client_requires_connection_and_valid_digit(self):
        with self.assertRaises(RuntimeError):
            await LiveKitTransportClient.send_dtmf(SimpleNamespace(_connected=False), "1")
        with self.assertRaises(ValueError):
            await LiveKitTransportClient.send_dtmf(SimpleNamespace(_connected=True), "x")
