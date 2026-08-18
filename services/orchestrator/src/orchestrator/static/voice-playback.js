(function (root, factory) {
  const playback = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = playback;
  } else {
    root.VoicePlayback = playback;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const DEFAULT_INITIAL_BUFFER_SECONDS = 0.96;
  const DEFAULT_CROSSFADE_SECONDS = 0.008;
  const DEFAULT_MINIMUM_LEAD_SECONDS = 0.04;
  const MAXIMUM_INITIAL_BUFFER_SECONDS = 1.44;
  const MAXIMUM_INITIAL_WAIT_SECONDS = 1.6;

  function recommendedInitialBufferSeconds(answerCharacterCount) {
    return Math.min(
      MAXIMUM_INITIAL_BUFFER_SECONDS,
      Math.max(DEFAULT_INITIAL_BUFFER_SECONDS, answerCharacterCount * 0.015),
    );
  }

  function voiceFailureDisposition(errorName, replyDisplayed) {
    if (errorName === "AbortError") return "interrupted";
    return replyDisplayed ? "playback_degraded" : "service_unavailable";
  }

  function chooseNonRepeatingAudioUrl(
    candidates,
    previousUrl,
    random = Math.random,
  ) {
    const unique = [...new Set((candidates || []).filter(Boolean))];
    if (!unique.length) return null;
    const selectable =
      unique.length > 1 ? unique.filter((url) => url !== previousUrl) : unique;
    const index = Math.min(
      selectable.length - 1,
      Math.floor(random() * selectable.length),
    );
    return selectable[index];
  }

  class VoicePlaybackScheduler {
    constructor({
      audioContext,
      destination,
      activeSources,
      initialBufferSeconds = DEFAULT_INITIAL_BUFFER_SECONDS,
      crossfadeSeconds = DEFAULT_CROSSFADE_SECONDS,
      minimumLeadSeconds = DEFAULT_MINIMUM_LEAD_SECONDS,
      now = () => performance.now(),
      setTimer = (callback, delayMs) => setTimeout(callback, delayMs),
      clearTimer = (timer) => clearTimeout(timer),
    }) {
      this.audioContext = audioContext;
      this.destination = destination;
      this.activeSources = activeSources;
      this.outputGain = audioContext.createGain();
      this.outputGain.gain.setValueAtTime(1, audioContext.currentTime);
      this.outputGain.connect(destination);
      this.outputVolume = 1;
      this.initialBufferSeconds = initialBufferSeconds;
      this.crossfadeSeconds = crossfadeSeconds;
      this.minimumLeadSeconds = minimumLeadSeconds;
      this.now = now;
      this.setTimer = setTimer;
      this.clearTimer = clearTimer;
      this.createdAtMs = now();
      this.contextStartedAt = audioContext.currentTime;
      this.queue = [];
      this.queuedDuration = 0;
      this.started = false;
      this.previousEndAt = null;
      this.lastPlayback = Promise.resolve();
      this.interrupted = false;
      this.interruptionReason = null;
      this.bargeInMetrics = null;
      this.chunkTimings = [];
      this.underrunCount = 0;
      this.underrunTotalMs = 0;
      this.underrunMaxMs = 0;
      this.initialBufferedMs = 0;
      this.firstPlaybackDelayMs = null;
      this.initialWaitExpired = false;
      this.startTimers = new Set();
      this.initialWaitTimer = setTimer(() => {
        this.initialWaitTimer = null;
        this.initialWaitExpired = true;
        if (this.queue.length) this._startQueuedPlayback();
      }, MAXIMUM_INITIAL_WAIT_SECONDS * 1000);
    }

    enqueue(audioBuffer, { arrivalTimeMs = this.now(), onStart = null } = {}) {
      if (!audioBuffer || !(audioBuffer.duration > 0)) return;
      const timing = {
        arrival_offset_ms: Math.max(0, arrivalTimeMs - this.createdAtMs),
        duration_ms: audioBuffer.duration * 1000,
        scheduled_start_offset_ms: null,
        gap_before_ms: 0,
      };
      this.chunkTimings.push(timing);

      if (!this.started) {
        this.queue.push({ audioBuffer, timing, onStart });
        this.queuedDuration += audioBuffer.duration;
        if (
          this.initialWaitExpired ||
          this.queuedDuration >= this.initialBufferSeconds
        ) {
          this._startQueuedPlayback();
        }
        return;
      }
      this._schedule(audioBuffer, timing, onStart);
    }

    async finish() {
      this._clearInitialWaitTimer();
      if (!this.started && this.queue.length) this._startQueuedPlayback();
      await this.lastPlayback;
      return this.metrics();
    }

    duck(volume = 0.15, fadeMs = 50) {
      this._setOutputVolume(volume, fadeMs);
    }

    restore(fadeMs = 50) {
      this._setOutputVolume(1, fadeMs);
    }

    recordBargeInMetrics(metrics) {
      this.bargeInMetrics = metrics ? { ...metrics } : null;
    }

    interrupt(reason = "manual", bargeInMetrics = null) {
      this.interrupted = true;
      this.interruptionReason = reason;
      if (bargeInMetrics) this.recordBargeInMetrics(bargeInMetrics);
      this._clearInitialWaitTimer();
      for (const source of [...this.activeSources]) {
        try {
          source.stop();
        } catch {}
      }
      this.activeSources.clear();
      for (const timer of this.startTimers) this.clearTimer(timer);
      this.startTimers.clear();
    }

    metrics() {
      return {
        chunk_count: this.chunkTimings.length,
        audio_duration_ms: this.chunkTimings.reduce(
          (total, chunk) => total + chunk.duration_ms,
          0,
        ),
        initial_buffered_ms: this.initialBufferedMs,
        first_playback_delay_ms: this.firstPlaybackDelayMs,
        buffer_target_ms: this.initialBufferSeconds * 1000,
        crossfade_ms: this.crossfadeSeconds * 1000,
        underrun_count: this.underrunCount,
        underrun_total_ms: this.underrunTotalMs,
        underrun_max_ms: this.underrunMaxMs,
        interrupted: this.interrupted,
        interruption_reason: this.interruptionReason,
        barge_in_mode: this.bargeInMetrics?.mode || null,
        barge_in_duck_latency_ms: this.bargeInMetrics?.duck_latency_ms ?? null,
        barge_in_confirm_latency_ms: this.bargeInMetrics?.confirm_latency_ms ?? null,
        barge_in_false_trigger_count:
          this.bargeInMetrics?.false_trigger_count || 0,
        chunk_timings: this.chunkTimings.map((timing) => ({ ...timing })),
      };
    }

    _setOutputVolume(volume, fadeMs) {
      const target = Math.max(0, Math.min(1, volume));
      const now = this.audioContext.currentTime;
      this.outputGain.gain.cancelScheduledValues?.(now);
      this.outputGain.gain.setValueAtTime(this.outputVolume, now);
      this.outputGain.gain.linearRampToValueAtTime(
        target,
        now + Math.max(0, fadeMs) / 1000,
      );
      this.outputVolume = target;
    }

    _startQueuedPlayback() {
      if (this.started || !this.queue.length) return;
      this._clearInitialWaitTimer();
      this.started = true;
      this.initialBufferedMs = this.queuedDuration * 1000;
      this.firstPlaybackDelayMs = this.now() - this.createdAtMs;
      const queued = this.queue;
      this.queue = [];
      this.queuedDuration = 0;
      for (const { audioBuffer, timing, onStart } of queued) {
        this._schedule(audioBuffer, timing, onStart);
      }
    }

    _clearInitialWaitTimer() {
      if (this.initialWaitTimer === null) return;
      this.clearTimer(this.initialWaitTimer);
      this.initialWaitTimer = null;
    }

    _schedule(audioBuffer, timing, onStart = null) {
      const now = this.audioContext.currentTime;
      const earliestStart = now + this.minimumLeadSeconds;
      let startAt = earliestStart;
      let gapSeconds = 0;
      if (this.previousEndAt !== null) {
        const desiredStart = this.previousEndAt - this.crossfadeSeconds;
        startAt = Math.max(desiredStart, earliestStart);
        gapSeconds = Math.max(0, startAt - this.previousEndAt);
      }

      if (gapSeconds > 0.0005) {
        const gapMs = gapSeconds * 1000;
        this.underrunCount += 1;
        this.underrunTotalMs += gapMs;
        this.underrunMaxMs = Math.max(this.underrunMaxMs, gapMs);
        timing.gap_before_ms = gapMs;
      }
      timing.scheduled_start_offset_ms =
        (startAt - this.contextStartedAt) * 1000;

      const source = this.audioContext.createBufferSource();
      const gain = this.audioContext.createGain();
      const endAt = startAt + audioBuffer.duration;
      const fadeSeconds = Math.min(
        this.crossfadeSeconds,
        audioBuffer.duration / 2,
      );
      source.buffer = audioBuffer;
      source.connect(gain);
      gain.connect(this.outputGain);
      gain.gain.setValueAtTime(0, startAt);
      gain.gain.linearRampToValueAtTime(1, startAt + fadeSeconds);
      gain.gain.setValueAtTime(1, endAt - fadeSeconds);
      gain.gain.linearRampToValueAtTime(0, endAt);

      this.lastPlayback = new Promise((resolve) => {
        source.onended = () => {
          this.activeSources.delete(source);
          try {
            source.disconnect();
            gain.disconnect();
          } catch {}
          resolve();
        };
      });
      this.activeSources.add(source);
      source.start(startAt);
      if (onStart) {
        const delayMs = Math.max(0, (startAt - this.audioContext.currentTime) * 1000);
        const timer = this.setTimer(() => {
          this.startTimers.delete(timer);
          if (!this.interrupted) onStart();
        }, delayMs);
        this.startTimers.add(timer);
      }
      this.previousEndAt = endAt;
    }
  }

  return {
    DEFAULT_CROSSFADE_SECONDS,
    DEFAULT_INITIAL_BUFFER_SECONDS,
    MAXIMUM_INITIAL_BUFFER_SECONDS,
    MAXIMUM_INITIAL_WAIT_SECONDS,
    VoicePlaybackScheduler,
    chooseNonRepeatingAudioUrl,
    recommendedInitialBufferSeconds,
    voiceFailureDisposition,
  };
});
