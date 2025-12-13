/**
 * AudioWorklet processor for capturing raw PCM audio.
 * Captures audio at native sample rate and resamples to 16kHz mono int16.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    // Target sample rate for ASR (16kHz)
    this.targetSampleRate = 16000;
    // Native sample rate (passed from main thread)
    this.nativeSampleRate = options.processorOptions?.sampleRate || 48000;
    // Resample ratio
    this.resampleRatio = this.nativeSampleRate / this.targetSampleRate;

    // Buffer to accumulate samples for resampling
    this.inputBuffer = [];

    // How many target samples to accumulate before sending (100ms = 1600 samples at 16kHz)
    this.chunkSize = 1600;
    this.outputBuffer = [];

    this.port.onmessage = (event) => {
      if (event.data.type === 'stop') {
        // Flush remaining samples
        if (this.outputBuffer.length > 0) {
          this.sendChunk(this.outputBuffer);
          this.outputBuffer = [];
        }
      }
    };
  }

  /**
   * Resample from native rate to 16kHz using linear interpolation.
   */
  resample(inputSamples) {
    if (this.resampleRatio === 1) {
      return inputSamples;
    }

    const outputLength = Math.floor(inputSamples.length / this.resampleRatio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * this.resampleRatio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, inputSamples.length - 1);
      const t = srcIndex - srcIndexFloor;

      // Linear interpolation
      output[i] = inputSamples[srcIndexFloor] * (1 - t) + inputSamples[srcIndexCeil] * t;
    }

    return output;
  }

  /**
   * Convert float32 samples to int16 and send to main thread.
   */
  sendChunk(samples) {
    // Convert float32 [-1, 1] to int16 [-32768, 32767]
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    // Send as ArrayBuffer (transferable)
    this.port.postMessage({
      type: 'audio',
      samples: int16.buffer
    }, [int16.buffer]);
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) {
      return true;
    }

    // Get mono channel (first channel)
    const monoInput = input[0];

    // Resample to 16kHz
    const resampled = this.resample(monoInput);

    // Accumulate in output buffer
    for (let i = 0; i < resampled.length; i++) {
      this.outputBuffer.push(resampled[i]);
    }

    // Send chunks when we have enough samples
    while (this.outputBuffer.length >= this.chunkSize) {
      const chunk = this.outputBuffer.slice(0, this.chunkSize);
      this.outputBuffer = this.outputBuffer.slice(this.chunkSize);
      this.sendChunk(chunk);
    }

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
