'use client';

import React, { useState } from 'react';
import { HelpCircle, ChevronDown, ChevronUp, Cpu, Volume2, ShieldCheck, Sparkles, AlertCircle } from 'lucide-react';

export const EducationalGuide: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'how_it_works' | 'ai_clues' | 'how_to_read'>('how_it_works');

  return (
    <div className="bg-[#0e1726]/90 border border-cyan-500/20 rounded-2xl p-5 shadow-lg space-y-4">
      {/* Header Banner Toggle */}
      <div 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between cursor-pointer select-none group"
      >
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-xl bg-cyan-950 border border-cyan-500/30 text-cyan-400 group-hover:scale-105 transition-transform">
            <HelpCircle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <span>New to Audio Forensics? Learn How AI Voice Detection Works</span>
              <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-[10px] font-mono">
                Easy Guide
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Click to explore simple explanations of voice features, AI artifacts, and audio graphs.
            </p>
          </div>
        </div>

        <button className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors">
          {isOpen ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
        </button>
      </div>

      {/* Expandable Content */}
      {isOpen && (
        <div className="pt-3 border-t border-slate-800 space-y-4 animate-fadeIn">
          {/* Tabs Navigation */}
          <div className="flex flex-wrap gap-2 border-b border-slate-800 pb-3">
            <button
              onClick={() => setActiveTab('how_it_works')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-2 transition-colors ${
                activeTab === 'how_it_works'
                  ? 'bg-cyan-600 text-slate-950 font-bold'
                  : 'bg-slate-900 text-slate-400 hover:text-white'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>1. How AI Cloning Works</span>
            </button>

            <button
              onClick={() => setActiveTab('ai_clues')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-2 transition-colors ${
                activeTab === 'ai_clues'
                  ? 'bg-cyan-600 text-slate-950 font-bold'
                  : 'bg-slate-900 text-slate-400 hover:text-white'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>2. Audio Clues VoxGuard Checks</span>
            </button>

            <button
              onClick={() => setActiveTab('how_to_read')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-2 transition-colors ${
                activeTab === 'how_to_read'
                  ? 'bg-cyan-600 text-slate-950 font-bold'
                  : 'bg-slate-900 text-slate-400 hover:text-white'
              }`}
            >
              <Volume2 className="w-3.5 h-3.5" />
              <span>3. How to Read Your Graphs</span>
            </button>
          </div>

          {/* Tab 1: How AI Cloning Works */}
          {activeTab === 'how_it_works' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300">
              <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="text-cyan-400 font-bold flex items-center space-x-1.5">
                  <span className="w-5 h-5 rounded-full bg-cyan-950 border border-cyan-500/30 flex items-center justify-center text-[10px]">1</span>
                  <span>Human Voice Production</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Real human speech comes from lungs pushing air through vocal cords and throat shapes. It includes natural pitch fluctuations, breathing pauses, and subtle physical vibrations.
                </p>
              </div>

              <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="text-cyan-400 font-bold flex items-center space-x-1.5">
                  <span className="w-5 h-5 rounded-full bg-cyan-950 border border-cyan-500/30 flex items-center justify-center text-[10px]">2</span>
                  <span>AI Voice Generator (TTS)</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  AI voice models generate mathematical sound waves based on learned text-to-speech algorithms. They recreate realistic words, but often struggle with micro-dynamics of human vocal cords.
                </p>
              </div>

              <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="text-cyan-400 font-bold flex items-center space-x-1.5">
                  <span className="w-5 h-5 rounded-full bg-cyan-950 border border-cyan-500/30 flex items-center justify-center text-[10px]">3</span>
                  <span>Digital Signal Inspection</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  VoxGuard analyzes the invisible sound frequencies, pitch smoothness, and high-frequency boundaries to highlight acoustic clues without making unverified assumptions.
                </p>
              </div>
            </div>
          )}

          {/* Tab 2: Audio Clues */}
          {activeTab === 'ai_clues' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300">
              <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="text-amber-400 font-bold flex items-center space-x-1.5">
                  <AlertCircle className="w-4 h-4" />
                  <span>Pitch Smoothness (Centroid)</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Human pitch naturally rises and falls with emotion and emphasis. AI voices can sometimes sound unnaturally robotic or overly uniform in tone over long sentences.
                </p>
              </div>

              <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="text-amber-400 font-bold flex items-center space-x-1.5">
                  <AlertCircle className="w-4 h-4" />
                  <span>Treble Boundary (Rolloff)</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Real microphone recordings capture ambient room reverberation up to high frequencies. Many AI generators cut off sharply at fixed frequency boundaries (e.g. 8 kHz or 12 kHz).
                </p>
              </div>

              <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="text-amber-400 font-bold flex items-center space-x-1.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <span>Vocal Texture (MFCC)</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Mel-Frequency Cepstral Coefficients (MFCCs) capture the unique resonance of a speaker's mouth and vocal tract. VoxGuard extracts these features for scientific verification.
                </p>
              </div>
            </div>
          )}

          {/* Tab 3: How to Read Your Graphs */}
          {activeTab === 'how_to_read' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300">
              <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="text-cyan-400 font-bold flex items-center space-x-2">
                  <Volume2 className="w-4 h-4" />
                  <span>Waveform (Loudness Envelope)</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  The green waveform shows the volume changes over time. Spikes represent loud speech sounds, while quiet gaps show pauses or breaths. Smooth, natural variations indicate normal speech dynamics.
                </p>
              </div>

              <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="text-cyan-400 font-bold flex items-center space-x-2">
                  <Cpu className="w-4 h-4" />
                  <span>Spectrogram (Frequency Heat Map)</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  The colored grid maps sound frequencies from low bass (bottom) to high treble (top). Bright colors (yellow/cyan) show strong voice energy. Dark areas show quiet frequencies.
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
