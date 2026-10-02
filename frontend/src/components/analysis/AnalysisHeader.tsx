'use client';

import React, { useState } from 'react';
import { RefreshCw, FileText, Download, Copy, Check } from 'lucide-react';
import { AnalysisResponse } from '@/lib/types';

interface AnalysisHeaderProps {
  filename: string;
  analysisResult: AnalysisResponse;
  onReset: () => void;
}

export const AnalysisHeader: React.FC<AnalysisHeaderProps> = ({ filename, analysisResult, onReset }) => {
  const [copied, setCopied] = useState<boolean>(false);

  const handleDownloadJson = () => {
    const jsonStr = JSON.stringify(analysisResult, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voxguard_report_${filename.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleCopySummary = () => {
    const summaryText = `[VoxGuard Forensic Report]\nFile: ${filename}\nStatus: ${analysisResult.classification.status}\nVerdict: ${analysisResult.classification.label}\nDuration: ${analysisResult.audio_metadata.duration_seconds.toFixed(2)}s\nSample Rate: ${analysisResult.audio_metadata.sample_rate} Hz Mono\nEngine: ${analysisResult.classification.model_name}`;
    navigator.clipboard.writeText(summaryText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
      <div>
        <div className="flex items-center space-x-2">
          <FileText className="w-5 h-5 text-cyan-400" />
          <h2 className="text-xl font-bold tracking-tight text-slate-100 font-mono truncate max-w-xl">
            {filename}
          </h2>
        </div>
        <p className="text-xs text-slate-400 font-mono mt-1">
          Forensic Signal Analysis & Acoustic Feature Inspection Report
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={handleCopySummary}
          className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 font-mono text-xs flex items-center space-x-1.5 transition-colors"
          title="Copy summary text"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-cyan-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
          <span>{copied ? 'Copied!' : 'Copy Summary'}</span>
        </button>

        <button
          onClick={handleDownloadJson}
          className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-cyan-300 font-mono text-xs flex items-center space-x-1.5 transition-colors"
          title="Download full JSON report"
        >
          <Download className="w-3.5 h-3.5 text-cyan-400" />
          <span>Export JSON</span>
        </button>

        <button
          onClick={onReset}
          className="px-3 py-1.5 rounded-lg bg-cyan-950/80 hover:bg-cyan-900/90 border border-cyan-500/40 text-cyan-300 font-mono text-xs flex items-center space-x-1.5 transition-colors font-semibold"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>New Analysis</span>
        </button>
      </div>
    </div>
  );
};
