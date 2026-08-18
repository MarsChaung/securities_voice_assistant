(function (root, factory) {
  const bargeIn = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = bargeIn;
  } else {
    root.VoiceBargeIn = bargeIn;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const NON_ACTIONABLE_UTTERANCE_RE =
    /^(?:嗯+(?:哼+)?|呃+|啊+|喔+|哦+|唔+|那個|那个|就是|好(?:的)?|對(?:啊)?|对(?:啊)?|(?:我)?(?:知道|了解|瞭解|明白)(?:了)?|收到(?:了)?|謝謝(?:你)?|谢谢(?:你)?|感謝(?:你)?|感谢(?:你)?)[，,、。！？!?\s]*$/u;

  function isNonActionableUtterance(text) {
    return NON_ACTIONABLE_UTTERANCE_RE.test((text || "").trim());
  }

  function sanitizeAsrTranscript(text) {
    return (text || "").replace(/\uFFFD/gu, "");
  }

  function hasMeaningfulTranscript(text) {
    return /[\p{L}\p{N}]/u.test(sanitizeAsrTranscript(text));
  }

  function shouldResumePlaybackAfterBargeIn(text) {
    const sanitized = sanitizeAsrTranscript(text).trim();
    if (!hasMeaningfulTranscript(sanitized)) return true;
    if (/^(?:再見|掰掰|拜拜)[，,、。！？!?\s]*$/u.test(sanitized)) return false;
    if (isNonActionableUtterance(sanitized)) return true;
    const meaningfulCharacters = sanitized.match(/[\p{L}\p{N}]/gu) || [];
    return meaningfulCharacters.length <= 2;
  }

  function compactContextEchoText(text) {
    return sanitizeAsrTranscript(text)
      .toLocaleLowerCase()
      .replace(/臺/gu, "台")
      .replace(/[^\p{L}\p{N}]/gu, "");
  }

  function isLikelyContextEcho(text, context) {
    const compactText = compactContextEchoText(text);
    const compactContext = compactContextEchoText(context);
    if (compactText.length < 16 || !compactContext) return false;
    if (compactContext.includes(compactText)) return true;

    const terms = [
      ...new Set(
        (context || "")
          .split(/[、,，；;\n]+/u)
          .map(compactContextEchoText)
          .filter((term) => term.length >= 2),
      ),
    ];
    const covered = Array(compactText.length).fill(false);
    let matchedTerms = 0;
    for (const term of terms) {
      let start = compactText.indexOf(term);
      if (start < 0) continue;
      matchedTerms += 1;
      while (start >= 0) {
        covered.fill(true, start, Math.min(start + term.length, covered.length));
        start = compactText.indexOf(term, start + 1);
      }
    }
    const coverage = covered.filter(Boolean).length / covered.length;
    return matchedTerms >= 4 && coverage >= 0.65;
  }

  class BargeInDetector {
    constructor({
      preset,
      now = () => performance.now(),
      onDuck = () => {},
      onRestore = () => {},
      onConfirm = () => {},
    }) {
      this.now = now;
      this.onDuck = onDuck;
      this.onRestore = onRestore;
      this.onConfirm = onConfirm;
      this.noiseFloorDb = -60;
      this.falseTriggerCount = 0;
      this.setPreset(preset);
      this.disarm({ restore: false });
    }

    setPreset(preset) {
      if (!preset) throw new Error("barge-in preset is required");
      this.preset = { ...preset };
    }

    arm() {
      this.armed = true;
      this.falseTriggerCount = 0;
      this._resetCandidate();
    }

    disarm({ restore = true } = {}) {
      if (restore && this.ducked) this.onRestore(this.metrics());
      this.armed = false;
      this._resetCandidate();
    }

    markSpeech(atMs = this.now()) {
      if (!this.armed || this.onsetAtMs === null) return;
      this.speechConfirmed = true;
      this.speechConfirmedAtMs = atMs;
    }

    processLevel(dbfs, atMs = this.now()) {
      if (!this.armed || !Number.isFinite(dbfs)) return;
      const thresholdDb = Math.max(
        this.preset.minimum_dbfs,
        this.noiseFloorDb + this.preset.energy_margin_db,
      );
      const aboveThreshold = dbfs >= thresholdDb;

      if (aboveThreshold) {
        if (this.onsetAtMs === null) {
          this.onsetAtMs = atMs;
          this.lastAboveAtMs = atMs;
        } else {
          this.lastAboveAtMs = atMs;
        }
        const elapsedMs = atMs - this.onsetAtMs;
        if (!this.ducked && elapsedMs >= this.preset.duck_after_ms) {
          this.ducked = true;
          this.duckAtMs = atMs;
          this.onDuck(this.metrics());
        }
        if (
          this.ducked
          && this.speechConfirmed
          && elapsedMs >= this.preset.confirm_ms
        ) {
          this.confirmAtMs = atMs;
          this.armed = false;
          this.onConfirm(this.metrics());
        }
        if (
          this.ducked
          && !this.speechConfirmed
          && elapsedMs
            >= this.preset.duck_after_ms + this.preset.false_trigger_timeout_ms
        ) {
          this.falseTriggerCount += 1;
          this.onRestore(this.metrics());
          this._resetCandidate();
        }
        return;
      }

      if (this.onsetAtMs === null) {
        this.noiseFloorDb = (this.noiseFloorDb * 0.98) + (dbfs * 0.02);
        return;
      }
      if (atMs - this.lastAboveAtMs < this.preset.false_trigger_timeout_ms) return;

      if (this.ducked) {
        this.falseTriggerCount += 1;
        this.onRestore(this.metrics());
      }
      this._resetCandidate();
    }

    metrics() {
      return {
        mode: this.preset.id,
        duck_latency_ms:
          this.duckAtMs === null || this.onsetAtMs === null
            ? null
            : this.duckAtMs - this.onsetAtMs,
        confirm_latency_ms:
          this.confirmAtMs === null || this.onsetAtMs === null
            ? null
            : this.confirmAtMs - this.onsetAtMs,
        false_trigger_count: this.falseTriggerCount,
      };
    }

    _resetCandidate() {
      this.onsetAtMs = null;
      this.lastAboveAtMs = null;
      this.duckAtMs = null;
      this.confirmAtMs = null;
      this.ducked = false;
      this.speechConfirmed = false;
      this.speechConfirmedAtMs = null;
    }
  }

  return {
    BargeInDetector,
    hasMeaningfulTranscript,
    isLikelyContextEcho,
    isNonActionableUtterance,
    shouldResumePlaybackAfterBargeIn,
    sanitizeAsrTranscript,
  };
});
