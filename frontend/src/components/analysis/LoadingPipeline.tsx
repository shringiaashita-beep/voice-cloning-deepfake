'use client';

import React, { useEffect, useState } from 'react';
import { Loader2, Check, Circle } from 'lucide-react';

const PIPELINE_STEPS = [
  'Validation',
  'Decoding',
  'Preprocessing',
  'Feature extraction',
  'Acoustic analysis',
  'Report generation'
];

export const LoadingPipeline: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<number>(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev < PIPELINE_STEPS.length - 1 ? prev + 1 : prev));
    }, 450);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="forensic-panel p-8 text-center max-w-xl mx-auto my-8 space-y-6">
      <div className="flex flex-col items-center justify-center space-y-3">
        <Loader2 className="w-9 h-9 text-cyan-400 animate-spin" />
        <h3 className="text-lg font-bold text-white tracking-tight">
          Analyzing audio
        </h3>
        <p className="text-xs text-slate-400 font-sans">
          Executing pipeline transformations and signal feature calculations...
        </p>
      </div>

      {/* Vertical / Horizontal Pipeline Steps */}
      <div className="space-y-2 font-mono text-xs text-left pt-2 border-t border-white/[0.08]">
        {PIPELINE_STEPS.map((step, idx) => {
          const isDone = idx < currentStep;
          const isCurrent = idx === currentStep;

          return (
            <div
              key={step}
              className={`p-2.5 rounded-xl border flex items-center justify-between transition-all ${
                isDone
                  ? 'bg-cyan-950/40 border-cyan-500/30 text-cyan-300'
                  : isCurrent
                  ? 'bg-slate-900 border-cyan-400 text-white font-bold'
                  : 'bg-[#0b0f17] border-white/[0.04] text-slate-500'
              }`}
            >
              <div className="flex items-center space-x-3">
                {isDone ? (
                  <Check className="w-4 h-4 text-cyan-400" />
                ) : isCurrent ? (
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                ) : (
                  <Circle className="w-3.5 h-3.5 text-slate-600" />
                )}
                <span>{step}</span>
              </div>
              <span className="text-[10px] text-slate-500">
                {isDone ? 'COMPLETE' : isCurrent ? 'RUNNING' : 'QUEUED'}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
