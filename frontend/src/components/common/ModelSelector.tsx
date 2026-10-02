'use client';

import React from 'react';
import { Shield, Sparkles, BarChart3, Check, Zap, Cpu } from 'lucide-react';

interface ModelSelectorProps {
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
  disabled?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onSelectModel,
  disabled = false,
}) => {
  return (
    <div className="bg-[#0b101b] border border-cyan-500/20 rounded-2xl p-4 shadow-lg space-y-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
        <div className="flex items-center space-x-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono">
            Forensic Inference Mode
          </span>
        </div>
        <div className="flex items-center space-x-2 text-[11px] font-mono">
          <span className="text-slate-400">Selected Engine:</span>
          <span className="text-cyan-300 font-bold px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/30">
            {selectedModel === 'ml_classifier'
              ? 'Active Neural Classifier (<50ms)'
              : selectedModel === 'baseline'
              ? 'Forensic Audit (Safe Baseline)'
              : 'Voice Clone Neural Ensemble'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Mode 1: Active Neural Classifier Mode (Pitch Highlight) */}
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSelectModel('ml_classifier')}
          className={`p-3.5 rounded-xl border text-left transition-all relative flex flex-col justify-between ${
            selectedModel === 'ml_classifier'
              ? 'bg-cyan-950/40 border-cyan-400 shadow-lg shadow-cyan-500/15 ring-1 ring-cyan-400'
              : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 opacity-80 hover:opacity-100'
          } ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
        >
          {selectedModel === 'ml_classifier' && (
            <div className="absolute top-3 right-3 w-5 h-5 rounded-full bg-cyan-400 text-slate-950 flex items-center justify-center">
              <Check className="w-3.5 h-3.5 stroke-[3]" />
            </div>
          )}
          <div>
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-bold text-white font-mono">Active Neural Classifier</span>
            </div>
            <div className="mt-1 flex items-center space-x-1.5">
              <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 text-[10px] font-mono font-bold">
                PACKAGED best_model.pt
              </span>
              <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-mono font-bold flex items-center space-x-1">
                <span>&lt;50ms</span>
              </span>
            </div>
            <p className="text-[11px] text-slate-300 mt-2 leading-snug font-sans">
              Loads trained PyTorch neural checkpoint (VoxGuardAcousticNet). Instant binary decision with calibrated synthetic confidence.
            </p>
          </div>
          <div className="mt-2.5 pt-2 border-t border-white/[0.06] text-[10px] text-cyan-400 font-mono flex items-center justify-between">
            <span>High-Impact Pitch Mode</span>
            <span>Latency ~13ms</span>
          </div>
        </button>

        {/* Mode 2: Forensic Audit Mode (Safe / Baseline) */}
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSelectModel('baseline')}
          className={`p-3.5 rounded-xl border text-left transition-all relative flex flex-col justify-between ${
            selectedModel === 'baseline'
              ? 'bg-amber-950/40 border-amber-400 shadow-lg shadow-amber-500/15 ring-1 ring-amber-400'
              : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 opacity-80 hover:opacity-100'
          } ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
        >
          {selectedModel === 'baseline' && (
            <div className="absolute top-3 right-3 w-5 h-5 rounded-full bg-amber-400 text-slate-950 flex items-center justify-center">
              <Check className="w-3.5 h-3.5 stroke-[3]" />
            </div>
          )}
          <div>
            <div className="flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-amber-400" />
              <span className="text-xs font-bold text-white font-mono">Forensic Audit Mode</span>
            </div>
            <div className="mt-1">
              <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[10px] font-mono font-bold">
                SAFE / BASELINE
              </span>
            </div>
            <p className="text-[11px] text-slate-300 mt-2 leading-snug font-sans">
              Extracts pure objective DSP signal indicators (spectral rolloff, pitch micro-jitter, spectral flux) with zero speculative inferences.
            </p>
          </div>
          <div className="mt-2.5 pt-2 border-t border-white/[0.06] text-[10px] text-amber-400 font-mono flex items-center justify-between">
            <span>Physical Acoustic Evidence</span>
            <span>Court-Admissible</span>
          </div>
        </button>

        {/* Mode 3: Voice Clone Neural Ensemble */}
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSelectModel('voice_clone_detector')}
          className={`p-3.5 rounded-xl border text-left transition-all relative flex flex-col justify-between ${
            selectedModel === 'voice_clone_detector'
              ? 'bg-purple-950/40 border-purple-400 shadow-lg shadow-purple-500/15 ring-1 ring-purple-400'
              : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 opacity-80 hover:opacity-100'
          } ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
        >
          {selectedModel === 'voice_clone_detector' && (
            <div className="absolute top-3 right-3 w-5 h-5 rounded-full bg-purple-400 text-slate-950 flex items-center justify-center">
              <Check className="w-3.5 h-3.5 stroke-[3]" />
            </div>
          )}
          <div>
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-purple-400" />
              <span className="text-xs font-bold text-white font-mono">Clone Architecture Ensemble</span>
            </div>
            <div className="mt-1">
              <span className="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 text-[10px] font-mono font-bold">
                MULTI-MODEL ENSEMBLE
              </span>
            </div>
            <p className="text-[11px] text-slate-300 mt-2 leading-snug font-sans">
              Multi-branch forensic attribution engine diagnosing specific vocoder fingerprints: ElevenLabs, RVC, XTTS, and Bark.
            </p>
          </div>
          <div className="mt-2.5 pt-2 border-t border-white/[0.06] text-[10px] text-purple-400 font-mono flex items-center justify-between">
            <span>Generator Attribution</span>
            <span>Fingerprint Matrix</span>
          </div>
        </button>
      </div>
    </div>
  );
};
