'use client';

import React from 'react';
import { CheckCircle2, Clock, Workflow } from 'lucide-react';
import { Classification } from '@/lib/types';

interface ProcessingPipelineProps {
  classification: Classification;
}

const PIPELINE_STAGES = [
  { id: 'upload', label: 'Upload', desc: 'Binary payload received' },
  { id: 'validation', label: 'Validation', desc: 'MIME & magic check' },
  { id: 'decode', label: 'Decode', desc: 'PCM float32 stream' },
  { id: 'preprocess', label: 'Preprocess', desc: '16kHz mono scaling' },
  { id: 'features', label: 'Feature Extraction', desc: 'Log-Mel & MFCC DSP' },
  { id: 'classification', label: 'Classification', desc: 'Baseline / ML Model' },
  { id: 'visualization', label: 'Visualization', desc: 'Peak & spectrogram grid' },
  { id: 'report', label: 'Forensic Report', desc: 'Provenance assembly' }
];

export const ProcessingPipeline: React.FC<ProcessingPipelineProps> = ({ classification }) => {
  const meta = classification.metadata || {};
  const execTime = meta.execution_time_ms ? `${meta.execution_time_ms.toFixed(2)} ms` : null;

  return (
    <div className="forensic-panel p-6 space-y-4">
      <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
        <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-slate-200 flex items-center space-x-2">
          <Workflow className="w-4 h-4 text-cyan-400" />
          <span>Processing Pipeline & Signal Transformation Timeline</span>
        </h3>
        {execTime && (
          <span className="flex items-center space-x-1.5 px-2.5 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/30 text-cyan-300 font-mono text-[10px] uppercase font-bold">
            <Clock className="w-3 h-3 text-cyan-400" />
            <span>Exec: {execTime}</span>
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
        {PIPELINE_STAGES.map((stage) => (
          <div
            key={stage.id}
            className="bg-[#0b0f17] p-3 rounded-xl border border-white/[0.06] space-y-1 relative"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-400 font-bold uppercase">{stage.label}</span>
              <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />
            </div>
            <p className="text-[11px] text-slate-300 truncate" title={stage.desc}>
              {stage.desc}
            </p>
            <div className="text-[9px] text-cyan-400/90 font-semibold tracking-wide uppercase pt-1">
              {stage.id === 'classification' && execTime ? `SUCCESS (${execTime})` : 'COMPLETE'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
