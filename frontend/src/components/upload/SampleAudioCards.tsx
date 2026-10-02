'use client';

import React from 'react';
import { Play, Sparkles, ShieldAlert, ShieldCheck, Cpu, Mic } from 'lucide-react';

export type SamplePreset = 'elevenlabs' | 'rvc' | 'bark' | 'human' | 'wav' | 'mp3' | 'ogg';

interface SampleAudioCardsProps {
  onSelectSample: (preset: SamplePreset, filename: string) => void;
  isLoading: boolean;
}

export const SampleAudioCards: React.FC<SampleAudioCardsProps> = ({ onSelectSample, isLoading }) => {
  return (
    <div className="space-y-4 pt-4 border-t border-white/[0.08]">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
              Voice Clone Test Bench Presets
            </h4>
          </div>
          <p className="text-xs text-slate-400 mt-0.5 font-sans">
            Benchmark VoxGuard against real synthetic voice cloning signatures vs authentic human acoustics.
          </p>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/30 text-cyan-300">
          One-Click Test
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 font-sans">
        {/* Sample 1: ElevenLabs Neural Vocoder */}
        <button
          onClick={() => onSelectSample('elevenlabs', 'elevenlabs_neural_clone.wav')}
          disabled={isLoading}
          className="forensic-panel p-4 text-left group hover:border-purple-500/50 hover:bg-purple-950/10 transition-all cursor-pointer flex flex-col justify-between min-h-[125px] relative overflow-hidden"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-purple-950/90 text-purple-300 border border-purple-500/30 flex items-center space-x-1">
              <Cpu className="w-3 h-3" />
              <span>ELEVENLABS</span>
            </span>
            <div className="w-7 h-7 rounded-full bg-slate-900 flex items-center justify-center text-purple-400 group-hover:bg-purple-600 group-hover:text-slate-950 transition-colors">
              <Play className="w-3.5 h-3.5 fill-current ml-0.5" />
            </div>
          </div>
          <div>
            <h5 className="text-xs font-bold text-slate-100 group-hover:text-purple-300 transition-colors">
              Neural Vocoder Clone
            </h5>
            <p className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-tight">
              7.5 kHz spectral cutoff & ultra-smooth micro-pitch signature.
            </p>
          </div>
          <div className="text-[10px] font-mono text-red-400 flex items-center space-x-1 pt-1 border-t border-white/[0.04]">
            <ShieldAlert className="w-3 h-3" />
            <span>Expected: AI Synthetic</span>
          </div>
        </button>

        {/* Sample 2: RVC Voice Conversion */}
        <button
          onClick={() => onSelectSample('rvc', 'rvc_pitch_shift_clone.wav')}
          disabled={isLoading}
          className="forensic-panel p-4 text-left group hover:border-amber-500/50 hover:bg-amber-950/10 transition-all cursor-pointer flex flex-col justify-between min-h-[125px] relative overflow-hidden"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-amber-950/90 text-amber-300 border border-amber-500/30 flex items-center space-x-1">
              <Cpu className="w-3 h-3" />
              <span>RVC v2</span>
            </span>
            <div className="w-7 h-7 rounded-full bg-slate-900 flex items-center justify-center text-amber-400 group-hover:bg-amber-600 group-hover:text-slate-950 transition-colors">
              <Play className="w-3.5 h-3.5 fill-current ml-0.5" />
            </div>
          </div>
          <div>
            <h5 className="text-xs font-bold text-slate-100 group-hover:text-amber-300 transition-colors">
              Voice Conversion Clone
            </h5>
            <p className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-tight">
              Discrete pitch quantization steps & formant shift warping.
            </p>
          </div>
          <div className="text-[10px] font-mono text-red-400 flex items-center space-x-1 pt-1 border-t border-white/[0.04]">
            <ShieldAlert className="w-3 h-3" />
            <span>Expected: AI Synthetic</span>
          </div>
        </button>

        {/* Sample 3: Bark / Diffusion TTS */}
        <button
          onClick={() => onSelectSample('bark', 'bark_diffusion_clone.wav')}
          disabled={isLoading}
          className="forensic-panel p-4 text-left group hover:border-red-500/50 hover:bg-red-950/10 transition-all cursor-pointer flex flex-col justify-between min-h-[125px] relative overflow-hidden"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-red-950/90 text-red-300 border border-red-500/30 flex items-center space-x-1">
              <Cpu className="w-3 h-3" />
              <span>BARK / XTTS</span>
            </span>
            <div className="w-7 h-7 rounded-full bg-slate-900 flex items-center justify-center text-red-400 group-hover:bg-red-600 group-hover:text-slate-950 transition-colors">
              <Play className="w-3.5 h-3.5 fill-current ml-0.5" />
            </div>
          </div>
          <div>
            <h5 className="text-xs font-bold text-slate-100 group-hover:text-red-300 transition-colors">
              Diffusion Token Speech
            </h5>
            <p className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-tight">
              Spectral flux jitter & synthetic latent pause dithering.
            </p>
          </div>
          <div className="text-[10px] font-mono text-red-400 flex items-center space-x-1 pt-1 border-t border-white/[0.04]">
            <ShieldAlert className="w-3 h-3" />
            <span>Expected: AI Synthetic</span>
          </div>
        </button>

        {/* Sample 4: Authentic Biological Speech */}
        <button
          onClick={() => onSelectSample('human', 'authentic_human_speech.wav')}
          disabled={isLoading}
          className="forensic-panel p-4 text-left group hover:border-emerald-500/50 hover:bg-emerald-950/10 transition-all cursor-pointer flex flex-col justify-between min-h-[125px] relative overflow-hidden"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-950/90 text-emerald-300 border border-emerald-500/30 flex items-center space-x-1">
              <Mic className="w-3 h-3" />
              <span>HUMAN</span>
            </span>
            <div className="w-7 h-7 rounded-full bg-slate-900 flex items-center justify-center text-emerald-400 group-hover:bg-emerald-600 group-hover:text-slate-950 transition-colors">
              <Play className="w-3.5 h-3.5 fill-current ml-0.5" />
            </div>
          </div>
          <div>
            <h5 className="text-xs font-bold text-slate-100 group-hover:text-emerald-300 transition-colors">
              Natural Human Voice
            </h5>
            <p className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-tight">
              Organic micro-jitter, natural vocal tract formants & breath noise.
            </p>
          </div>
          <div className="text-[10px] font-mono text-emerald-400 flex items-center space-x-1 pt-1 border-t border-white/[0.04]">
            <ShieldCheck className="w-3 h-3" />
            <span>Expected: Human Voice</span>
          </div>
        </button>
      </div>
    </div>
  );
};
