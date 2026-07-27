const test = require("node:test");
const assert = require("node:assert/strict");

const {
  VoicePlaybackScheduler,
  recommendedInitialBufferSeconds,
} = require("../../services/orchestrator/src/orchestrator/static/voice-playback.js");

class FakeAudioParam {
  constructor() {
    this.events = [];
  }

  setValueAtTime(value, at) {
    this.events.push(["set", value, at]);
  }

  linearRampToValueAtTime(value, at) {
    this.events.push(["ramp", value, at]);
  }
}

class FakeGain {
  constructor() {
    this.gain = new FakeAudioParam();
  }

  connect() {}

  disconnect() {}
}

class FakeSource {
  constructor() {
    this.buffer = null;
    this.onended = null;
    this.startedAt = null;
  }

  connect() {}

  disconnect() {}

  start(at) {
    this.startedAt = at;
    queueMicrotask(() => this.onended?.());
  }

  stop() {
    queueMicrotask(() => this.onended?.());
  }
}

class FakeAudioContext {
  constructor() {
    this.currentTime = 0;
    this.destination = {};
    this.sources = [];
    this.gains = [];
  }

  createBufferSource() {
    const source = new FakeSource();
    this.sources.push(source);
    return source;
  }

  createGain() {
    const gain = new FakeGain();
    this.gains.push(gain);
    return gain;
  }
}

test("scales the initial buffer for long answers and caps the delay", () => {
  assert.equal(recommendedInitialBufferSeconds(20), 1.2);
  assert.equal(recommendedInitialBufferSeconds(50), 2);
  assert.equal(recommendedInitialBufferSeconds(100), 3);
  assert.equal(recommendedInitialBufferSeconds(1000), 3);
});

test("starts buffered audio when the 2.8 second wait limit is reached", async () => {
  const context = new FakeAudioContext();
  let nowMs = 0;
  let initialWaitCallback = null;
  const scheduler = new VoicePlaybackScheduler({
    audioContext: context,
    destination: context.destination,
    activeSources: new Set(),
    initialBufferSeconds: 3,
    now: () => nowMs,
    setTimer: (callback) => {
      initialWaitCallback = callback;
      return 1;
    },
    clearTimer: () => {},
  });

  scheduler.enqueue({ duration: 0.5 });
  assert.equal(context.sources.length, 0);

  context.currentTime = 2.8;
  nowMs = 2800;
  initialWaitCallback();
  const metrics = await scheduler.finish();

  assert.equal(context.sources.length, 1);
  assert.equal(metrics.first_playback_delay_ms, 2800);
});

test("starts the first audio immediately when it arrives after the wait limit", async () => {
  const context = new FakeAudioContext();
  let initialWaitCallback = null;
  const scheduler = new VoicePlaybackScheduler({
    audioContext: context,
    destination: context.destination,
    activeSources: new Set(),
    initialBufferSeconds: 3,
    now: () => 3000,
    setTimer: (callback) => {
      initialWaitCallback = callback;
      return 1;
    },
    clearTimer: () => {},
  });

  initialWaitCallback();
  context.currentTime = 3;
  scheduler.enqueue({ duration: 0.5 });
  await scheduler.finish();

  assert.equal(context.sources.length, 1);
});

test("buffers 1.2 seconds and crossfades adjacent chunks without gaps", async () => {
  const context = new FakeAudioContext();
  const activeSources = new Set();
  let nowMs = 0;
  const scheduler = new VoicePlaybackScheduler({
    audioContext: context,
    destination: context.destination,
    activeSources,
    now: () => nowMs,
  });

  context.currentTime = 0.2;
  nowMs = 200;
  scheduler.enqueue({ duration: 0.5 }, { arrivalTimeMs: nowMs });
  context.currentTime = 0.7;
  nowMs = 700;
  scheduler.enqueue({ duration: 0.5 }, { arrivalTimeMs: nowMs });
  assert.equal(context.sources.length, 0);

  context.currentTime = 1.1;
  nowMs = 1100;
  scheduler.enqueue({ duration: 0.5 }, { arrivalTimeMs: nowMs });
  const metrics = await scheduler.finish();

  assert.equal(context.sources.length, 3);
  assert.equal(metrics.initial_buffered_ms, 1500);
  assert.equal(metrics.underrun_count, 0);
  assert.equal(context.sources[1].startedAt - context.sources[0].startedAt, 0.492);
  assert.deepEqual(context.gains[1].gain.events.map((event) => event[0]), [
    "set",
    "ramp",
    "set",
    "ramp",
  ]);
});

test("records a real late-arrival gap after playback has started", async () => {
  const context = new FakeAudioContext();
  let nowMs = 0;
  const scheduler = new VoicePlaybackScheduler({
    audioContext: context,
    destination: context.destination,
    activeSources: new Set(),
    now: () => nowMs,
  });

  context.currentTime = 0;
  scheduler.enqueue({ duration: 0.6 });
  scheduler.enqueue({ duration: 0.6 });
  context.currentTime = 1.3;
  nowMs = 1300;
  scheduler.enqueue({ duration: 0.6 });

  const metrics = await scheduler.finish();
  assert.equal(metrics.underrun_count, 1);
  assert.ok(metrics.underrun_total_ms > 100);
  assert.equal(metrics.chunk_timings[2].gap_before_ms, metrics.underrun_max_ms);
});

test("flushes a short response when the stream ends", async () => {
  const context = new FakeAudioContext();
  const scheduler = new VoicePlaybackScheduler({
    audioContext: context,
    destination: context.destination,
    activeSources: new Set(),
    now: () => 50,
  });

  scheduler.enqueue({ duration: 0.4 });
  assert.equal(context.sources.length, 0);

  const metrics = await scheduler.finish();
  assert.equal(context.sources.length, 1);
  assert.equal(metrics.initial_buffered_ms, 400);
  assert.equal(metrics.chunk_count, 1);
});
