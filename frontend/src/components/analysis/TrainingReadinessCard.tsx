'use client';

import React, { useState, useEffect } from 'react';
import { Database, CheckCircle, AlertTriangle, ChevronDown, ChevronUp, Layers, HardDrive, ShieldCheck } from 'lucide-react';
import { fetchTrainingStatus } from '@/lib/api';
import { TrainingStatusResponse } from '@/lib/types';

export const TrainingReadinessCard: React.FC = () => {
  const [statusData, setStatusData] = useState<TrainingStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isPipelineExpanded, setIsPipelineExpanded] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    async function loadStatus() {
      try {
        const data = await fetchTrainingStatus();
        if (isMounted) {
          setStatusData(data);
          setIsLoading(false);
        }
      } catch {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }
    loadStatus();
    return () => { isMounted = false; };
  }, []);

  if (isLoading) {
    return (
      <div className="forensic-card rounded-xl p-6 text-center font-mono text-xs text-slate-400 animate-pulse">
        Loading Training Data Status...
      </div>
    );
  }

  if (!statusData) {
    return (
      <div className="forensic-card rounded-xl p-6 text-center font-mono text-xs text-amber-400">
        Training data status is unavailable. No readiness claims can be made.
      </div>
    );
  }

  const totalWindows = statusData.class_balance.human + statusData.class_balance.synthetic || statusData.windows;
  const humanPct = totalWindows > 0 ? ((statusData.class_balance.human / totalWindows) * 100).toFixed(1) : '50.0';
  const synthPct = totalWindows > 0 ? ((statusData.class_balance.synthetic / totalWindows) * 100).toFixed(1) : '50.0';

  return (
    <div className="forensic-card rounded-xl p-6 space-y-5">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2.5">
          <Database className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold font-mono uppercase tracking-wider text-slate-200">
            TRAINING DATA STATUS
          </h3>
        </div>

        <div className="flex items-center space-x-2 font-mono text-xs">
          <span className="text-slate-400">Dataset Validation:</span>
          <span className={`px-2.5 py-0.5 rounded-full border text-[11px] font-bold uppercase ${
            statusData.training_ready
              ? 'bg-emerald-950/80 border-emerald-500/40 text-emerald-400'
              : 'bg-amber-950/80 border-amber-500/40 text-amber-400'
          }`}>
            {statusData.training_ready ? 'READY' : 'NOT READY'}
          </span>
        </div>
      </div>

      {/* Grid Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 font-mono text-xs">
        {/* Metric 1: Feature Artifacts */}
        <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 space-y-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider">Feature Artifacts</div>
          <div className="text-base font-bold text-slate-100">{statusData.feature_artifacts.toLocaleString()}</div>
          <div className="text-[10px] text-cyan-400/80 truncate">Compressed .npz</div>
        </div>

        {/* Metric 2: Training Windows */}
        <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 space-y-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider">Training Windows</div>
          <div className="text-base font-bold text-slate-100">{statusData.windows.toLocaleString()}</div>
          <div className="text-[10px] text-slate-400">256 frames (stride 128)</div>
        </div>

        {/* Metric 3: Speakers */}
        <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 space-y-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider">Unique Speakers</div>
          <div className="text-base font-bold text-slate-100">{statusData.speakers}</div>
          <div className="text-[10px] text-emerald-400">Speaker-Disjoint</div>
        </div>

        {/* Metric 4: Class Balance */}
        <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 space-y-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider">Class Balance</div>
          <div className="text-xs font-bold text-slate-100 flex justify-between pt-0.5">
            <span className="text-emerald-400">H: {humanPct}%</span>
            <span className="text-red-400">S: {synthPct}%</span>
          </div>
          <div className="text-[10px] text-slate-400">Human / Synthetic</div>
        </div>

        {/* Metric 5: Speaker Leakage */}
        <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 space-y-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider">Speaker Leakage</div>
          <div className="flex items-center space-x-1 pt-0.5">
            {!statusData.speaker_leakage ? (
              <span className="text-emerald-400 font-bold flex items-center space-x-1">
                <CheckCircle className="w-3.5 h-3.5" /> <span>PASS</span>
              </span>
            ) : (
              <span className="text-red-400 font-bold flex items-center space-x-1">
                <AlertTriangle className="w-3.5 h-3.5" /> <span>FAIL</span>
              </span>
            )}
          </div>
          <div className="text-[10px] text-slate-400">Zero Overlap Check</div>
        </div>

        {/* Metric 6: Feature Integrity */}
        <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 space-y-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider">Feature Integrity</div>
          <div className="flex items-center space-x-1 pt-0.5">
            {statusData.feature_integrity ? (
              <span className="text-emerald-400 font-bold flex items-center space-x-1">
                <CheckCircle className="w-3.5 h-3.5" /> <span>PASS</span>
              </span>
            ) : (
              <span className="text-amber-400 font-bold flex items-center space-x-1">
                <AlertTriangle className="w-3.5 h-3.5" /> <span>FAIL</span>
              </span>
            )}
          </div>
          <div className="text-[10px] text-slate-400">No NaN / Inf arrays</div>
        </div>

        {/* Metric 7: OOD Split */}
        <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 space-y-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider">OOD Split</div>
          <div className="text-xs font-bold pt-0.5">
            {statusData.ood_available ? (
              <span className="text-cyan-400">AVAILABLE</span>
            ) : (
              <span className="text-amber-400">NOT AVAILABLE</span>
            )}
          </div>
          <div className="text-[10px] text-slate-400">Out-of-Distribution</div>
        </div>

        {/* Metric 8: Training Readiness */}
        <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800/80 space-y-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider">Training Readiness</div>
          <div className="text-xs font-bold pt-0.5">
            {statusData.training_ready ? (
              <span className="text-emerald-400">READY</span>
            ) : (
              <span className="text-amber-400">BLOCKED</span>
            )}
          </div>
          <div className="text-[10px] text-slate-400">Structural Pass</div>
        </div>
      </div>

      {/* Structural Pass Notice */}
      <div className="p-3 rounded-lg bg-cyan-950/30 border border-cyan-500/20 text-[11px] font-sans text-slate-300 flex items-start space-x-2">
        <ShieldCheck className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
        <span>
          <strong>Structural Readiness Notice:</strong> "READY" status confirms the local dataset passed structural, windowing, and speaker-disjoint validation checks. It does NOT indicate that an active ML detection model is loaded into production inference.
        </span>
      </div>

      {statusData.validation_errors && statusData.validation_errors.length > 0 && (
        <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-500/20 text-[11px] font-sans text-amber-200">
          <strong>Validation blockers:</strong>
          <ul className="mt-1 list-disc list-inside space-y-1">
            {statusData.validation_errors.slice(0, 3).map((error) => (
              <li key={error}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Expandable Training Pipeline Diagram */}
      <div className="border-t border-slate-800 pt-3">
        <button
          onClick={() => setIsPipelineExpanded(!isPipelineExpanded)}
          className="w-full flex items-center justify-between text-xs font-mono text-slate-400 hover:text-slate-200 transition-colors"
        >
          <span className="flex items-center space-x-2">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span>Training Infrastructure Pipeline Diagram</span>
          </span>
          {isPipelineExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {isPipelineExpanded && (
          <div className="mt-3 p-4 rounded-xl bg-slate-950/90 border border-slate-800 font-mono text-[11px] text-slate-300 animate-fadeIn">
            <div className="flex flex-wrap items-center justify-center gap-2 text-center">
              <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300">Manifest CSV</span>
              <span className="text-cyan-400">➔</span>
              <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300">Speaker Split</span>
              <span className="text-cyan-400">➔</span>
              <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300">Feature Generation</span>
              <span className="text-cyan-400">➔</span>
              <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300">Feature Validation</span>
              <span className="text-cyan-400">➔</span>
              <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300">256-Frame Windowing</span>
              <span className="text-cyan-400">➔</span>
              <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300">Batch Construction</span>
              <span className="text-cyan-400">➔</span>
              <span className="px-2.5 py-1 rounded bg-emerald-950 border border-emerald-500/40 text-emerald-400 font-bold">Training Ready</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
