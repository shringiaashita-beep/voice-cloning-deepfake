/**
 * Audio Degradation Engine for VoxGuard.
 * Simulates real-world lossy transmission channels using Web Audio API OfflineAudioContext:
 * 1. WhatsApp Voice Note (Opus ~16kbps, high-frequency cutoff at 6.8kHz, acoustic compression)
 * 2. 8kHz PSTN / Cellular Phone Line (Bandpass 300Hz - 3400Hz, downsampled to 8kHz, phone noise floor)
 */

export type DegradationMode = 'none' | 'whatsapp' | 'phone_8k';

export interface DegradationProfile {
  id: DegradationMode;
  name: string;
  badge: string;
  description: string;
  technicalDetails: string;
}

export const DEGRADATION_PROFILES: DegradationProfile[] = [
  {
    id: 'none',
    name: 'Clean Direct Audio',
    badge: 'PRISTINE',
    description: 'Original high-fidelity audio without channel degradation.',
    technicalDetails: 'Full Nyquist bandwidth, 0 dB lossy compression.',
  },
  {
    id: 'whatsapp',
    name: 'WhatsApp Voice Note',
    badge: 'OPUS 16KBPS',
    description: 'Simulates WhatsApp Opus audio compression with 6.8kHz cutoff.',
    technicalDetails: 'Low-pass filter @ 6800 Hz (-24 dB/oct), acoustic compression curve.',
  },
  {
    id: 'phone_8k',
    name: '8kHz Phone Line',
    badge: 'PSTN / 8KHZ',
    description: 'Simulates cellular / landline telephone speech band (AMR/G.711 narrowband).',
    technicalDetails: 'Bandpass 300 Hz - 3400 Hz, 8kHz downsampling, companded dynamic range.',
  },
];

/**
 * Encodes an AudioBuffer into standard 16-bit PCM WAV bytes.
 */
function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const numChannels = 1; // mono for forensic consistency
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;

  const channelData = buffer.getChannelData(0);
  const dataLength = channelData.length * (bitDepth / 8);
  const headerLength = 44;
  const totalLength = headerLength + dataLength;

  const arrayBuffer = new ArrayBuffer(totalLength);
  const view = new DataView(arrayBuffer);

  // RIFF identifier
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, 'WAVE');

  // "fmt " sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, format, true); // AudioFormat PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true); // ByteRate
  view.setUint16(32, numChannels * (bitDepth / 8), true); // BlockAlign
  view.setUint16(34, bitDepth, true);

  // "data" sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);

  // Write PCM samples with soft clipping
  let offset = 44;
  for (let i = 0; i < channelData.length; i++) {
    const s = Math.max(-1, Math.min(1, channelData[i]));
    const val = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, val, true);
    offset += 2;
  }

  return arrayBuffer;
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

/**
 * Applies degradation filters to an audio File using OfflineAudioContext.
 * Returns a new degraded File with appropriate suffix.
 */
export async function applyDegradationFilter(
  file: File,
  mode: DegradationMode
): Promise<{ file: File; duration: number }> {
  if (mode === 'none') {
    return { file, duration: 0 };
  }

  const arrayBuffer = await file.arrayBuffer();
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const tempCtx = new AudioCtx();
  const decodedBuffer = await tempCtx.decodeAudioData(arrayBuffer);
  await tempCtx.close();

  const duration = decodedBuffer.duration;
  const targetSampleRate = mode === 'phone_8k' ? 8000 : 16000;
  const offlineCtx = new OfflineAudioContext(1, Math.ceil(duration * targetSampleRate), targetSampleRate);

  // Buffer source
  const source = offlineCtx.createBufferSource();
  source.buffer = decodedBuffer;

  if (mode === 'whatsapp') {
    // WhatsApp voice note: low-pass at 6800 Hz + compressor
    const lowpass = offlineCtx.createBiquadFilter();
    lowpass.type = 'lowpass';
    lowpass.frequency.value = 6800;
    lowpass.Q.value = 0.7;

    const compressor = offlineCtx.createDynamicsCompressor();
    compressor.threshold.value = -24;
    compressor.knee.value = 12;
    compressor.ratio.value = 8;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.25;

    source.connect(lowpass);
    lowpass.connect(compressor);
    compressor.connect(offlineCtx.destination);
  } else if (mode === 'phone_8k') {
    // 8kHz PSTN phone line: bandpass 300Hz - 3400Hz + compander
    const highpass = offlineCtx.createBiquadFilter();
    highpass.type = 'highpass';
    highpass.frequency.value = 300;
    highpass.Q.value = 0.8;

    const lowpass = offlineCtx.createBiquadFilter();
    lowpass.type = 'lowpass';
    lowpass.frequency.value = 3400;
    lowpass.Q.value = 0.8;

    const compressor = offlineCtx.createDynamicsCompressor();
    compressor.threshold.value = -18;
    compressor.knee.value = 8;
    compressor.ratio.value = 6;
    compressor.attack.value = 0.005;
    compressor.release.value = 0.15;

    source.connect(highpass);
    highpass.connect(lowpass);
    lowpass.connect(compressor);
    compressor.connect(offlineCtx.destination);
  }

  source.start(0);
  const renderedBuffer = await offlineCtx.startRendering();
  const wavBytes = audioBufferToWav(renderedBuffer);

  const baseName = file.name.replace(/\.[^/.]+$/, '');
  const degradedFileName = `${baseName}_${mode}.wav`;
  const degradedFile = new File([wavBytes], degradedFileName, { type: 'audio/wav' });

  return { file: degradedFile, duration };
}
