import { SamplePreset } from '@/components/upload/SampleAudioCards';

/**
 * In-memory realistic audio synthesizer for test presets and forensic comparisons.
 * Generates valid RIFF/WAV 16-bit PCM audio buffers simulating distinctive vocoder,
 * conversion, diffusion, and human acoustic signatures.
 */
export function createSampleAudioFile(preset: SamplePreset, name: string): File {
  const duration = 2.0;
  const sampleRate = 16000;
  const numFrames = Math.floor(duration * sampleRate);
  const dataSize = numFrames * 2;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);

  // RIFF header
  view.setUint32(0, 0x52494646, false); // "RIFF"
  view.setUint32(4, 36 + dataSize, true);
  view.setUint32(8, 0x57415645, false); // "WAVE"
  view.setUint32(12, 0x666d7420, false); // "fmt "
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  view.setUint32(36, 0x64617461, false); // "data"
  view.setUint32(40, dataSize, true);

  if (preset === 'elevenlabs') {
    // 1. ElevenLabs Neural Vocoder Simulation:
    // Pure harmonic speech with 0 micro-jitter and sharp cutoff above 3.5 kHz (no high frequencies)
    const baseFreq = 210.0;
    const harmonics = [1, 2, 3, 4, 5, 7, 9, 12, 15]; // Up to ~3150 Hz
    for (let i = 0; i < numFrames; i++) {
      const t = i / sampleRate;
      let val = 0;
      for (const h of harmonics) {
        val += (1.0 / Math.sqrt(h)) * Math.sin(2 * Math.PI * (baseFreq * h) * t);
      }
      const env = Math.sin(Math.PI * (i / numFrames));
      const sample = Math.max(-1, Math.min(1, val * 0.25 * env)) * 32767;
      view.setInt16(44 + i * 2, sample, true);
    }
  } else if (preset === 'rvc') {
    // 2. RVC Voice Conversion Simulation:
    // Discrete pitch quantization steps (jumps between 220Hz, 260Hz, 330Hz) and formant warping
    let phase = 0;
    for (let i = 0; i < numFrames; i++) {
      const t = i / sampleRate;
      const noteIdx = Math.floor(t * 3.5) % 3;
      const f = noteIdx === 0 ? 220 : noteIdx === 1 ? 260 : 330;
      phase += (2 * Math.PI * f) / sampleRate;
      const val = Math.sin(phase) + 0.6 * Math.sin(2 * phase) + 0.4 * Math.sin(3 * phase);
      const env = Math.sin(Math.PI * (i / numFrames));
      const sample = Math.max(-1, Math.min(1, val * 0.3 * env)) * 32767;
      view.setInt16(44 + i * 2, sample, true);
    }
  } else if (preset === 'bark') {
    // 3. Bark / Diffusion Simulation:
    // Elevated spectral flux and token dithering
    let phase = 0;
    for (let i = 0; i < numFrames; i++) {
      const t = i / sampleRate;
      const tokenJitter = Math.sin(2 * Math.PI * 40 * t) > 0 ? 1.05 : 0.95;
      phase += (2 * Math.PI * 180 * tokenJitter) / sampleRate;
      const dither = (Math.random() - 0.5) * 0.08;
      const val = Math.sin(phase) + 0.5 * Math.sin(2 * phase) + dither;
      const env = Math.sin(Math.PI * (i / numFrames));
      const sample = Math.max(-1, Math.min(1, val * 0.35 * env)) * 32767;
      view.setInt16(44 + i * 2, sample, true);
    }
  } else {
    // 4. Authentic Human Voice Simulation:
    // Organic vocal fold micro-jitter (~1.2%), rich formants (F1-F4) spanning full spectrum up to 7.5 kHz, natural breath aspiration
    let phase = 0;
    for (let i = 0; i < numFrames; i++) {
      const t = i / sampleRate;
      const jitter = 1.0 + 0.018 * Math.sin(2 * Math.PI * 4.5 * t) + (Math.random() - 0.5) * 0.008;
      phase += (2 * Math.PI * 145 * jitter) / sampleRate;
      const val =
        0.5 * Math.sin(phase) +
        0.35 * Math.sin(2 * phase) +
        0.25 * Math.sin(3 * phase) +
        0.2 * Math.sin(5 * phase) +
        0.15 * Math.sin(8 * phase) +
        0.1 * Math.sin(14 * phase) +
        0.08 * Math.sin(20 * phase) +
        0.05 * Math.sin(28 * phase) +
        (Math.random() - 0.5) * 0.05; // natural breath aspiration
      const env = Math.sin(Math.PI * (i / numFrames));
      const sample = Math.max(-1, Math.min(1, val * 0.3 * env)) * 32767;
      view.setInt16(44 + i * 2, sample, true);
    }
  }

  const content = new Uint8Array(buf);
  return new File([content.buffer], name, { type: 'audio/wav' });
}
