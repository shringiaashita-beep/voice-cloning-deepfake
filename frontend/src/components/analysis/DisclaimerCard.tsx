import React from 'react';
import { ShieldAlert, BookOpen } from 'lucide-react';

interface DisclaimerCardProps {
  disclaimerText: string;
}

export const DisclaimerCard: React.FC<DisclaimerCardProps> = ({ disclaimerText }) => {
  const cleanedText = disclaimerText
    ? disclaimerText.replace(/\s*Refer to docs\/EVALUATION_METRICS\.md for benchmarking protocols\.\s*/gi, '')
    : '';

  return (
    <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 text-xs font-mono text-slate-400">
      <div className="flex items-start space-x-3">
        <ShieldAlert className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1.5">
          <h4 className="text-slate-200 font-bold uppercase tracking-wider text-[11px]">
            Mandatory Forensic Disclaimer & Policy Notice
          </h4>
          <p className="leading-relaxed text-slate-400">
            {cleanedText}
          </p>
          <div className="pt-1 flex items-center space-x-2 text-[11px] text-cyan-400 font-medium">
            <BookOpen className="w-3.5 h-3.5 flex-shrink-0" />
            <span>Refer to docs/EVALUATION_METRICS.md for benchmark standards and leakage protocols.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
