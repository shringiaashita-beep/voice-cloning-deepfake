import React from 'react';
import { Activity, BarChart2, Zap, Sliders } from 'lucide-react';
import { ForensicReportPayload } from '@/lib/types';

interface AcousticObservationsProps {
  observations: ForensicReportPayload['acoustic_observations'];
}

export const AcousticObservations: React.FC<AcousticObservationsProps> = ({ observations }) => {
  const centroid = observations?.spectral_centroid;
  const rolloff = observations?.spectral_rolloff_85_percent;
  const mfcc = observations?.mfcc_statistics;
  const signal = observations?.signal_characteristics;

  return (
    <div className="forensic-card rounded-xl p-6">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
        <h3 className="text-sm font-semibold font-mono uppercase tracking-wider text-slate-200 flex items-center space-x-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span>Objective Acoustic Signal Observations</span>
        </h3>
        <span className="text-xs text-slate-400 font-mono">DSP Measurements</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Spectral Centroid Box */}
        <div className="bg-slate-950/60 p-4 rounded-lg border border-slate-800/60 font-mono space-y-2">
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-semibold uppercase">
            <BarChart2 className="w-4 h-4" />
            <span>Voice Pitch Sharpness (Spectral Centroid)</span>
          </div>
          {centroid ? (
            <div className="space-y-1.5 text-xs text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">Average Pitch:</span>
                <span className="font-bold text-slate-100">{centroid.mean_hz.toFixed(1)} Hz</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Pitch Naturalness (Variation):</span>
                <span>{centroid.std_hz.toFixed(1)} Hz</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Pitch Range:</span>
                <span>{centroid.min_hz.toFixed(0)} Hz – {centroid.max_hz.toFixed(0)} Hz</span>
              </div>
              <div className="text-[11px] text-cyan-300/80 pt-1.5 border-t border-white/[0.06] font-sans">
                💡 <strong>Simple Note:</strong> Measures how bright or sharp the voice sounds. Real voices fluctuate naturally.
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">N/A</p>
          )}
        </div>

        {/* Spectral Rolloff Box */}
        <div className="bg-slate-950/60 p-4 rounded-lg border border-slate-800/60 font-mono space-y-2">
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-semibold uppercase">
            <Zap className="w-4 h-4" />
            <span>Treble Limit (Spectral Rolloff 85%)</span>
          </div>
          {rolloff ? (
            <div className="space-y-1.5 text-xs text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">High Frequency Boundary:</span>
                <span className="font-bold text-slate-100">{rolloff.mean_hz.toFixed(1)} Hz</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Boundary Variation:</span>
                <span>{rolloff.std_hz.toFixed(1)} Hz</span>
              </div>
              <div className="text-[11px] text-cyan-300/80 pt-1.5 border-t border-white/[0.06] font-sans">
                💡 <strong>Simple Note:</strong> Shows where 85% of sound energy stops. AI voices often cut off abruptly.
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">N/A</p>
          )}
        </div>

        {/* MFCC Statistics Box */}
        <div className="bg-slate-950/60 p-4 rounded-lg border border-slate-800/60 font-mono space-y-2">
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-semibold uppercase">
            <Sliders className="w-4 h-4" />
            <span>Vocal Texture & Shape (MFCC)</span>
          </div>
          {mfcc ? (
            <div className="space-y-1.5 text-xs text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">Feature Coefficients:</span>
                <span className="font-bold text-slate-100">{mfcc.num_coefficients}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Average Vocal Energy:</span>
                <span>{mfcc.mean.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Texture Diversity:</span>
                <span>{mfcc.variance.toFixed(2)}</span>
              </div>
              <div className="text-[11px] text-cyan-300/80 pt-1.5 border-t border-white/[0.06] font-sans">
                💡 <strong>Simple Note:</strong> Analyzes the unique resonance created by throat and mouth shapes during speech.
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">N/A</p>
          )}
        </div>

        {/* Signal Characteristics Box */}
        <div className="bg-slate-950/60 p-4 rounded-lg border border-slate-800/60 font-mono space-y-2">
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-semibold uppercase">
            <Activity className="w-4 h-4" />
            <span>Loudness & Volume Range</span>
          </div>
          {signal ? (
            <div className="space-y-1.5 text-xs text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">Average Energy Level:</span>
                <span className="font-bold text-slate-100">{signal.average_energy_db.toFixed(1)} dB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Quiet to Loud Range:</span>
                <span>{signal.dynamic_range_db.toFixed(1)} dB</span>
              </div>
              <div className="text-[11px] text-cyan-300/80 pt-1.5 border-t border-white/[0.06] font-sans">
                💡 <strong>Simple Note:</strong> Measures how clear and dynamic the recording is from soft whispers to loud speech.
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">N/A</p>
          )}
        </div>
      </div>
    </div>
  );
};
