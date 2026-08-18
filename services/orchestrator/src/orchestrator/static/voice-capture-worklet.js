class VoiceCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.blockSize = Math.max(128, Math.round(sampleRate * 0.03));
    this.samples = new Float32Array(this.blockSize);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    let sourceOffset = 0;
    while (sourceOffset < channel.length) {
      const available = this.blockSize - this.offset;
      const count = Math.min(available, channel.length - sourceOffset);
      this.samples.set(channel.subarray(sourceOffset, sourceOffset + count), this.offset);
      this.offset += count;
      sourceOffset += count;

      if (this.offset === this.blockSize) {
        let squareTotal = 0;
        for (const sample of this.samples) squareTotal += sample * sample;
        const rms = Math.sqrt(squareTotal / this.samples.length);
        const dbfs = rms > 0 ? 20 * Math.log10(rms) : -100;
        const completed = this.samples;
        this.port.postMessage(
          {
            type: "audio-frame",
            samples: completed,
            dbfs,
            duration_ms: (completed.length / sampleRate) * 1000,
          },
          [completed.buffer],
        );
        this.samples = new Float32Array(this.blockSize);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("voice-capture-processor", VoiceCaptureProcessor);
