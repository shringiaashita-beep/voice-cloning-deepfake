import React from 'react';
import { Classification } from '@/lib/types';

interface StatusCardProps {
  classification: Classification;
}

export const StatusCard: React.FC<StatusCardProps> = ({ classification }) => {
  const status = classification.status || 'analysis_only';
  const isReady = status === 'ready';
  const isNotReady = status === 'not_ready';

  const formatProb = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return 'N/A';
    return `${(val * 100).toFixed(1)}%`;
  };

  const humanProbStr = formatProb(classification.probabilities?.human);
  const syntheticProbStr = formatProb(classification.probabilities?.synthetic);
  const confidenceStr = formatProb(classification.confidence_score);

  let borderTheme = 'border-l-amber-500';
  let badgeTheme = 'bg-amber-950/80 border-amber-500/40 text-amber-300';
  let statusTextTheme = 'text-amber-400';
  let statusLabel = 'ANALYSIS ONLY';
  let headingText = 'INCONCLUSIVE';
  let descriptionText =
    '🔬 Audio Analysis Complete: We have extracted your recording\'s sound wave patterns, pitch stability, and frequency signals. Since no trained AI classifier is active, VoxGuard provides objective acoustic measurements without fake probability scores.';

  if (isNotReady) {
    statusLabel = 'MODEL NOT READY';
    headingText = 'INCONCLUSIVE';
    descriptionText =
      'Configured model weights are unavailable or failed validation checks. VoxGuard is operating in baseline observation mode.';
  } else if (isReady) {
    statusLabel = 'MODEL READY';
    badgeTheme = 'bg-cyan-950/80 border-cyan-500/40 text-cyan-300';
    statusTextTheme = 'text-cyan-400';

    if (classification.label === 'human') {
      borderTheme = 'border-l-emerald-500';
      headingText = 'REAL HUMAN VOICE DETECTED';
      descriptionText = `Evaluated AI classifier (${classification.model_name}) detected natural human speech patterns in this recording.`;
    } else if (classification.label === 'synthetic') {
      borderTheme = 'border-l-red-500';
      headingText = 'AI-GENERATED VOICE DETECTED';
      descriptionText = `Evaluated AI classifier (${classification.model_name}) detected synthetic/AI voice patterns in this recording.`;
    } else {
      borderTheme = 'border-l-amber-500';
      headingText = 'UNCERTAIN VERDICT';
      descriptionText = `Evaluated AI classifier (${classification.model_name}) produced an inconclusive decision score.`;
    }
  }

  return (
    <div className={`forensic-panel p-6 border-l-4 ${borderTheme} space-y-4`}>
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-2">
          <div className="flex items-center space-x-2.5">
            <span className={`text-[11px] font-mono font-bold tracking-wider ${statusTextTheme} uppercase`}>
              ANALYSIS STATUS & VERDICT
            </span>
            <span className={`px-2.5 py-0.5 rounded-full border font-mono text-[10px] font-bold uppercase ${badgeTheme}`}>
              {statusLabel}
            </span>
          </div>

          <h2 className="text-2xl font-bold text-white tracking-tight">
            {headingText}
          </h2>

          <p className="text-xs text-slate-300 max-w-3xl leading-relaxed font-sans">
            {descriptionText}
          </p>
        </div>

        {/* Classification Metrics Box */}
        <div className="bg-[#0b0f17] border border-white/10 rounded-xl p-4 font-mono text-xs text-right min-w-[250px]">
          <div className="text-slate-400 mb-2 font-semibold text-[11px] uppercase tracking-wider">
            Classification Summary
          </div>
          <div className="flex justify-between py-1 border-b border-white/[0.06] text-slate-400">
            <span>Human Probability:</span>
            <span className={`font-bold ${isReady ? 'text-emerald-400' : 'text-slate-400'}`}>
              {humanProbStr}
            </span>
          </div>
          <div className="flex justify-between py-1 border-b border-white/[0.06] text-slate-400">
            <span>Synthetic Probability:</span>
            <span className={`font-bold ${isReady ? 'text-red-400' : 'text-slate-400'}`}>
              {syntheticProbStr}
            </span>
          </div>
          <div className="flex justify-between py-1 text-slate-400">
            <span>Model Confidence:</span>
            <span className={`font-bold ${isReady ? 'text-cyan-300' : 'text-slate-400'}`}>
              {confidenceStr}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
