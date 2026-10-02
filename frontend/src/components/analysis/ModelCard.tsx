import React from 'react';
import { Cpu } from 'lucide-react';
import { Classification } from '@/lib/types';

interface ModelCardProps {
  classification: Classification;
}

export const ModelCard: React.FC<ModelCardProps> = ({ classification }) => {
  const isReady = classification.status === 'ready';

  const formatProb = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return 'N/A';
    return `${(val * 100).toFixed(1)}%`;
  };

  const confidenceDisplay = isReady ? formatProb(classification.confidence_score) : 'N/A';
  const probDisplay = isReady
    ? classification.probabilities?.synthetic != null
      ? `Synth: ${formatProb(classification.probabilities.synthetic)}`
      : 'N/A'
    : 'N/A';

  const statusBadgeStyle = isReady
    ? 'bg-cyan-950/80 border-cyan-500/40 text-cyan-300'
    : 'bg-amber-950/80 border-amber-500/40 text-amber-300';

  return (
    <div className="forensic-panel p-6 space-y-4">
      <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
        <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-slate-200 flex items-center space-x-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span>Active Classification Model</span>
        </h3>
        <span className={`px-2 py-0.5 rounded border font-mono text-[10px] uppercase font-bold ${statusBadgeStyle}`}>
          {classification.status.replace('_', ' ')}
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">MODEL</span>
          <span className="text-white font-bold text-xs leading-snug break-words block" title={classification.model_name}>
            {classification.model_name}
          </span>
        </div>

        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">VERSION</span>
          <span className="text-cyan-300 font-bold text-xs">
            {classification.model_version}
          </span>
        </div>

        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">CONFIDENCE</span>
          <span className={`font-bold text-xs ${isReady ? 'text-cyan-300' : 'text-slate-400'}`}>
            {confidenceDisplay}
          </span>
        </div>

        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">PROBABILITY</span>
          <span className={`font-bold text-xs ${isReady ? 'text-red-400' : 'text-slate-400'}`}>
            {probDisplay}
          </span>
        </div>
      </div>
    </div>
  );
};
