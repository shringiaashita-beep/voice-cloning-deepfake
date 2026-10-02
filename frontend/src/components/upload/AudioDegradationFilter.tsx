'use client';

import React from 'react';
import { Smartphone, Radio, Sparkles, Activity, Check } from 'lucide-react';
import { DegradationMode, DEGRADATION_PROFILES } from '@/lib/audioDegradation';

interface AudioDegradationFilterProps {
  selectedFilter: DegradationMode;
  onSelectFilter: (mode: DegradationMode) => void;
  disabled?: boolean;
}

export const AudioDegradationFilter: React.FC<AudioDegradationFilterProps> = ({
  selectedFilter,
  onSelectFilter,
  disabled = false,
}) => {
  return (
    <div className="bg-[#0b101c] border border-cyan-500/20 rounded-2xl p-4 shadow-lg space-y-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono">
            Acoustic Degradation Filter (Forensic Stress-Test)
          </h4>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          Simulate real-world lossy codecs & channel downsampling
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
        {DEGRADATION_PROFILES.map((profile) => {
          const isSelected = selectedFilter === profile.id;
          const Icon = profile.id === 'whatsapp' ? Smartphone : (profile.id === 'phone_8k' ? Radio : Sparkles);
          
          return (
            <button
              key={profile.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelectFilter(profile.id)}
              className={`p-3 rounded-xl border text-left transition-all relative flex flex-col justify-between ${
                isSelected
                  ? 'bg-cyan-950/40 border-cyan-500/70 shadow-md shadow-cyan-500/10'
                  : 'bg-slate-900/50 border-slate-800/80 hover:border-slate-700 opacity-80 hover:opacity-100'
              } ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
            >
              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Icon className={`w-3.5 h-3.5 ${isSelected ? 'text-cyan-400' : 'text-slate-400'}`} />
                    <span className="text-xs font-bold text-white font-mono">{profile.name}</span>
                  </div>
                  {isSelected && (
                    <div className="w-4 h-4 rounded-full bg-cyan-500 text-slate-950 flex items-center justify-center">
                      <Check className="w-3 h-3 stroke-[3]" />
                    </div>
                  )}
                </div>

                <div className="mt-1">
                  <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-mono font-bold ${
                    isSelected ? 'bg-cyan-500/20 text-cyan-300' : 'bg-slate-800 text-slate-400'
                  }`}>
                    {profile.badge}
                  </span>
                </div>

                <p className="text-[11px] text-slate-300 mt-1.5 leading-snug font-sans">
                  {profile.description}
                </p>
              </div>

              <div className="mt-2 pt-2 border-t border-white/[0.06] text-[10px] text-slate-400 font-mono">
                {profile.technicalDetails}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
