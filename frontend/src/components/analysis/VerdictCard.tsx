import React from 'react';
import { AlertCircle, Activity, Info } from 'lucide-react';
import { Classification } from '@/lib/types';

interface VerdictCardProps {
  classification: Classification;
}

export const VerdictCard: React.FC<VerdictCardProps> = ({ classification }) => {
  const isAnalysisOnly = classification.status === 'analysis_only';

  return (
    <div className="forensic-card rounded-xl p-6 border-l-4 border-l-amber-500">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <span className="px-2.5 py-1 rounded-full bg-amber-950/60 border border-amber-500/40 text-amber-300 font-mono text-xs font-semibold uppercase tracking-wider flex items-center space-x-1.5">
              <Activity className="w-3.5 h-3.5" />
              <span>Mode: Signal Analysis Only</span>
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Model: {classification.model_name} (v{classification.model_version})
            </span>
          </div>

          <h3 className="text-2xl font-bold text-slate-100 tracking-tight">
            Uncertain / Inconclusive Analysis
          </h3>
          <p className="text-sm text-slate-400 mt-1 max-w-3xl">
            VoxGuard is currently operating on the <strong className="text-slate-200">Statistical Acoustic Baseline</strong>. Objective physical signal characteristics are compiled below. No trained ML model weights are loaded.
          </p>
        </div>

        {/* Probabilities Box (Strictly Null) */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 font-mono text-xs text-right min-w-[220px]">
          <div className="text-slate-400 mb-1 flex items-center justify-end space-x-1">
            <Info className="w-3.5 h-3.5 text-slate-500" />
            <span>Classification Probabilities</span>
          </div>
          <div className="flex justify-between py-1 border-b border-slate-800 text-slate-400">
            <span>Human:</span>
            <span className="text-slate-400 font-semibold">null</span>
          </div>
          <div className="flex justify-between py-1 text-slate-400">
            <span>Synthetic:</span>
            <span className="text-slate-400 font-semibold">null</span>
          </div>
          <div className="mt-2 text-[10px] text-amber-400/90 tracking-tight">
            Uncalibrated Baseline (No ML Scores)
          </div>
        </div>
      </div>
    </div>
  );
};
