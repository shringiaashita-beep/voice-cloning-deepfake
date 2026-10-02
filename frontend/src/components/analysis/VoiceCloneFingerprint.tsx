'use client';

import React from 'react';
import { Fingerprint, CheckCircle2, AlertOctagon, Sparkles, Layers, Sliders } from 'lucide-react';
import { Classification } from '@/lib/types';

interface VoiceCloneFingerprintProps {
  classification: Classification;
}

export const VoiceCloneFingerprint: React.FC<VoiceCloneFingerprintProps> = ({ classification }) => {
  const meta = classification.metadata || {};
  const isReady = classification.status === 'ready';
  const scores = meta.architecture_scores || null;

  const dominantArch = meta.voice_clone_architecture || (isReady ? 'Neural Forensic Profile' : 'Not Available');
  const isSynthetic = classification.label === 'synthetic';

  const formatScore = (val: number | undefined | null): string => {
    if (val === undefined || val === null || !isReady) return 'N/A';
    return `${val.toFixed(1)}% match`;
  };

  const architectures = [
    {
      id: 'elevenlabs',
      name: 'ElevenLabs / Neural Vocoder (HiFi-GAN Profile)',
      scoreVal: scores?.elevenlabs_neural_vocoder,
      description: 'Ultra-smooth micro-pitch, steep 7.4 kHz high-frequency cutoff shelf, pristine harmonic purity.',
      color: 'from-purple-500 to-indigo-600',
      textColor: 'text-purple-400',
      badgeBg: 'bg-purple-950/80 border-purple-500/30',
    },
    {
      id: 'rvc',
      name: 'RVC / So-VITS Voice Conversion Profile',
      scoreVal: scores?.rvc_voice_conversion,
      description: 'Formant warping, pitch quantization steps, phase smearing at consonant transitions.',
      color: 'from-amber-500 to-orange-600',
      textColor: 'text-amber-400',
      badgeBg: 'bg-amber-950/80 border-amber-500/30',
    },
    {
      id: 'bark_diffusion',
      name: 'Diffusion & Autoregressive TTS (Bark / XTTS Profile)',
      scoreVal: scores?.bark_diffusion_tts,
      description: 'Acoustic token dithering, elevated frame-to-frame spectral flux, synthetic pause floor.',
      color: 'from-red-500 to-pink-600',
      textColor: 'text-red-400',
      badgeBg: 'bg-red-950/80 border-red-500/30',
    },
    {
      id: 'human',
      name: 'Human Vocal-Trait Similarity',
      scoreVal: scores?.natural_human_vocal_tract,
      description: 'Acoustic feature similarity measuring organic vocal fold micro-tremor (jitter 0.6-1.8%) and MFCC resonance.',
      color: 'from-emerald-500 to-teal-600',
      textColor: 'text-emerald-400',
      badgeBg: 'bg-emerald-950/80 border-emerald-500/30',
    },
  ];

  return (
    <div className="forensic-card rounded-2xl p-6 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center space-x-2.5">
            <Fingerprint className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-white tracking-tight">
              Voice Cloning Architecture Fingerprint
            </h3>
            <span className="px-2.5 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/30 text-cyan-300 font-mono text-[10px] uppercase font-bold">
              Experimental / Research Prototype
            </span>
          </div>
          <p className="text-xs text-slate-300 mt-1 font-sans">
            Compares acoustic spectral features against known voice synthesis profiles and biological vocal traits.
          </p>
        </div>

        <div className="flex items-center space-x-2 bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-slate-400 font-mono">Dominant Profile:</span>
          <span className={`text-xs font-mono font-bold ${isSynthetic ? 'text-red-400' : 'text-emerald-400'}`}>
            {dominantArch}
          </span>
        </div>
      </div>

      {/* Architecture Affinity Bars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {architectures.map((arch) => (
          <div
            key={arch.id}
            className="p-4 rounded-xl bg-[#090d16] border border-slate-800/90 hover:border-slate-700 transition-colors space-y-3"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className={`text-xs font-bold font-mono ${arch.textColor}`}>
                  {arch.name}
                </span>
              </div>
              <span className={`text-xs font-mono font-extrabold px-2 py-0.5 rounded-md border ${arch.badgeBg} ${arch.textColor}`}>
                {formatScore(arch.scoreVal)}
              </span>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
              <div
                className={`h-full bg-gradient-to-r ${arch.color} rounded-full transition-all duration-1000 ease-out`}
                style={{ width: `${arch.scoreVal !== undefined && arch.scoreVal !== null ? Math.min(100, Math.max(2, arch.scoreVal)) : 0}%` }}
              />
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed font-sans">
              {arch.description}
            </p>
          </div>
        ))}
      </div>

      {/* Forensic Clues Callout */}
      <div className="p-4 rounded-xl bg-cyan-950/30 border border-cyan-500/20 flex items-start space-x-3 text-xs text-slate-300">
        <Sparkles className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-bold text-cyan-300">Acoustic Feature Similarity vs Classification Probability:</span>
          <p className="leading-relaxed">
            Human Vocal-Trait Similarity measures signal/acoustic feature pattern alignment and is separate from the top-level classifier probability. Architecture affinities represent experimental research prototype feature similarity rather than definitive telemetry attribution.
          </p>
        </div>
      </div>
    </div>
  );
};
