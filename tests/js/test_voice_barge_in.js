const test = require("node:test");
const assert = require("node:assert/strict");

const {
  BargeInDetector,
  isNonActionableUtterance,
} = require("../../services/orchestrator/src/orchestrator/static/voice-barge-in.js");

const preset = {
  id: "standard",
  duck_after_ms: 100,
  confirm_ms: 250,
  energy_margin_db: 14,
  minimum_dbfs: -45,
  false_trigger_timeout_ms: 400,
};

test("recognises non-actionable turns without hiding a real follow-up question", () => {
  for (const text of [
    "嗯。",
    "嗯哼。",
    "呃！",
    "喔",
    "好的，",
    "我知道了。",
    "了解了",
    "瞭解了",
    "收到",
    "謝謝你",
  ]) {
    assert.equal(isNonActionableUtterance(text), true);
  }
  assert.equal(isNonActionableUtterance("嗯，我想問線上開戶資格"), false);
  assert.equal(isNonActionableUtterance("我知道了，那要如何補件？"), false);
  assert.equal(isNonActionableUtterance("好，請說明假除權息"), false);
});

test("ducks early and confirms only after server VAD confirms speech", () => {
  const events = [];
  const detector = new BargeInDetector({
    preset,
    onDuck: (metrics) => events.push(["duck", metrics]),
    onConfirm: (metrics) => events.push(["confirm", metrics]),
  });
  detector.arm();

  detector.processLevel(-20, 0);
  detector.processLevel(-20, 100);
  detector.processLevel(-20, 200);
  assert.deepEqual(events.map(([name]) => name), ["duck"]);

  detector.markSpeech(210);
  detector.processLevel(-20, 250);

  assert.deepEqual(events.map(([name]) => name), ["duck", "confirm"]);
  assert.equal(events[1][1].duck_latency_ms, 100);
  assert.equal(events[1][1].confirm_latency_ms, 250);
});

test("restores playback after a short false trigger", () => {
  const events = [];
  const detector = new BargeInDetector({
    preset,
    onDuck: () => events.push("duck"),
    onRestore: () => events.push("restore"),
    onConfirm: () => events.push("confirm"),
  });
  detector.arm();

  detector.processLevel(-20, 0);
  detector.processLevel(-20, 100);
  detector.processLevel(-80, 300);
  detector.processLevel(-80, 501);

  assert.deepEqual(events, ["duck", "restore"]);
  assert.equal(detector.metrics().false_trigger_count, 1);
});

test("does not duck distant speech below the adaptive threshold", () => {
  const events = [];
  const detector = new BargeInDetector({
    preset,
    onDuck: () => events.push("duck"),
  });
  detector.arm();

  detector.processLevel(-55, 0);
  detector.processLevel(-55, 200);
  detector.processLevel(-55, 500);

  assert.deepEqual(events, []);
});

test("restores playback when loud non-speech never receives VAD confirmation", () => {
  const events = [];
  const detector = new BargeInDetector({
    preset,
    onDuck: () => events.push("duck"),
    onRestore: () => events.push("restore"),
  });
  detector.arm();

  detector.processLevel(-20, 0);
  detector.processLevel(-20, 100);
  detector.processLevel(-20, 500);

  assert.deepEqual(events, ["duck", "restore"]);
  assert.equal(detector.metrics().false_trigger_count, 1);
});
