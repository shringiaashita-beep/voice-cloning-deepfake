import React from 'react';
import { Cpu, Clock, GitBranch } from 'lucide-react';
import { Classification } from '@/lib/types';

interface ProcessingStatsProps {
  classification: Classification;
}

export const ProcessingStats: React.FC<ProcessingStatsProps> = ({ classification }) => {
  const meta = classification.metadata || {};
  const execTime = meta.execution_time_ms ? `${meta.execution_time_ms} ms` : 'N/A';
  const device = meta.device ? meta.device.toUpperCase() : 'CPU';

  return (
    <div className="forensic-card rounded-xl p-4 font-mono text-xs text-slate-400">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-2">
          <Clock className="w-4 h-4 text-cyan-400" />
          <span>Execution Time: <strong className="text-slate-200">{execTime}</strong></span>
        </div>

        <div className="flex items-center space-x-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span>Compute Device: <strong className="text-slate-200">{device}</strong></span>
        </div>

        <div className="flex items-center space-x-2">
          <GitBranch className="w-4 h-4 text-cyan-400" />
          <span>Engine: <strong className="text-slate-200">{classification.model_name} (v{classification.model_version})</strong></span>
        </div>
      </div>
    </div>
  );
};
