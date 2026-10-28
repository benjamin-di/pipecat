#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Continuous output retains a sample clock when native buffering is discarded."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.transports.livekit.transport import LiveKitOutputTransport, LiveKitParams


class Clock:
    def __init__(self):
        self.now = 10.0

    def monotonic(self):
        return self.now

    async def sleep(self, delay):
        self.now += delay


class NativeQueue:
    """A bounded native source whose cleared samples cannot backpressure a writer."""

    def __init__(self, clock):
        self.clock = clock
        self.queued_until = clock.now
        self.clear_count = 0
        self.writes = 0

    def clear(self):
        self.queued_until = self.clock.now
        self.clear_count += 1

    async def publish_audio(self, frame):
        duration = frame.samples_per_channel / frame.sample_rate
        await self.clock.sleep(max(0, self.queued_until + duration - 0.25 - self.clock.now))
        self.queued_until = max(self.queued_until, self.clock.now) + duration
        self.clock.now += 0.001  # Native completion and Python scheduling overhead.
        self.writes += 1
        return True


class TestLiveKitOutputClock(unittest.IsolatedAsyncioTestCase):
    def output(self, clock, *, pacing=True):
        client = NativeQueue(clock)
        output = LiveKitOutputTransport(
            MagicMock(), client, LiveKitParams(audio_out_realtime_pacing=pacing)
        )
        output._sample_rate = 24000
        return output, client

    async def test_repeated_native_queue_clears_do_not_advance_the_sample_clock(self):
        clock = Clock()
        output, native = self.output(clock)
        frame = OutputAudioRawFrame(audio=bytes(960), sample_rate=24000, num_channels=1)
        offsets = []
        with (
            patch("pipecat.transports.livekit.transport.time", clock, create=True),
            patch("pipecat.transports.livekit.transport.asyncio.sleep", clock.sleep),
        ):
            for index in range(6000):
                if index % 500 == 0:
                    native.clear()
                self.assertTrue(await output.write_audio_frame(frame))
                if index % 500 == 499:
                    offsets.append((index + 1) * 0.02 - (clock.now - 10))
        self.assertEqual(native.clear_count, 12)
        self.assertLess(max(offsets) - min(offsets), 0.001)
        self.assertLess(abs(offsets[-1]), 0.025)

    async def test_a_sparse_stream_resumes_without_replaying_the_idle_interval(self):
        clock = Clock()
        output, native = self.output(clock)
        frame = OutputAudioRawFrame(audio=bytes(960), sample_rate=24000, num_channels=1)
        with (
            patch("pipecat.transports.livekit.transport.time", clock, create=True),
            patch("pipecat.transports.livekit.transport.asyncio.sleep", clock.sleep),
        ):
            await output.write_audio_frame(frame)
            clock.now += 30
            resumed = clock.now
            await output.write_audio_frame(frame)
            self.assertLess(clock.now - resumed, 0.005)
            await output.write_audio_frame(frame)
            self.assertAlmostEqual(clock.now - resumed, 0.021)
        self.assertEqual(native.writes, 3)

    async def test_cancelling_a_wait_does_not_publish_or_reserve_another_frame(self):
        clock = Clock()
        output, native = self.output(clock)
        frame = OutputAudioRawFrame(audio=bytes(960), sample_rate=24000, num_channels=1)
        with (
            patch("pipecat.transports.livekit.transport.time", clock, create=True),
            patch("pipecat.transports.livekit.transport.asyncio.sleep", clock.sleep),
        ):
            await output.write_audio_frame(frame)
            reserved = output._audio_next_write

            async def cancelled(delay):
                raise asyncio.CancelledError

            with patch("pipecat.transports.livekit.transport.asyncio.sleep", cancelled):
                with self.assertRaises(asyncio.CancelledError):
                    await output.write_audio_frame(frame)
            self.assertEqual(native.writes, 1)
            self.assertEqual(output._audio_next_write, reserved)
            await output.write_audio_frame(frame)
        self.assertEqual(native.writes, 2)

    async def test_pacing_is_opt_in(self):
        clock = Clock()
        output, _ = self.output(clock, pacing=False)
        with (
            patch("pipecat.transports.livekit.transport.time", clock, create=True),
            patch("pipecat.transports.livekit.transport.asyncio.sleep", clock.sleep),
        ):
            await output.write_audio_frame(
                OutputAudioRawFrame(audio=bytes(960), sample_rate=24000, num_channels=1)
            )
        self.assertIsNone(output._audio_next_write)
