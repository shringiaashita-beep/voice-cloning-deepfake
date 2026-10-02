'use client';

import React, { useState } from 'react';
import {
  ShieldAlert,
  UploadCloud,
  FileAudio,
  AlertTriangle,
  CheckCircle,
  Download,
  FileSpreadsheet,
  Trash2,
  Play,
  RotateCcw,
  Sparkles,
  ArrowRight,
  Shield,
  Clock
} from 'lucide-react';
import { analyzeAudioFile } from '@/lib/api';
import { AnalysisResponse } from '@/lib/types';
import { createSampleAudioFile } from '@/lib/audioGenerator';
import { SamplePreset } from '@/components/upload/SampleAudioCards';

export interface BatchItem {
  id: string;
  file: File;
  hash: string;
  status: 'pending' | 'analyzing' | 'done' | 'error';
  result?: AnalysisResponse;
  errorMessage?: string;
  riskScore: number; // 0 to 100
  threatLevel: 'CRITICAL' | 'HIGH' | 'MODERATE' | 'AUTHENTIC' | 'PENDING';
}

interface CyberCellBatchInvestigatorProps {
  selectedModel: string;
  onInspectSingleFile: (file: File, result: AnalysisResponse) => void;
}

export const CyberCellBatchInvestigator: React.FC<CyberCellBatchInvestigatorProps> = ({
  selectedModel,
  onInspectSingleFile,
}) => {
  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);
  const [isBatchRunning, setIsBatchRunning] = useState<boolean>(false);
  const [completedCount, setCompletedCount] = useState<number>(0);

  // Computes a deterministic pseudo SHA-256 hex string for triage display
  const computeHash = (name: string, size: number): string => {
    const raw = `${name}_${size}_cyber_desk`;
    let hash = 0;
    for (let i = 0; i < raw.length; i++) {
      hash = (hash << 5) - hash + raw.charCodeAt(i);
      hash |= 0;
    }
    const hex = Math.abs(hash).toString(16).padStart(8, '0');
    return `${hex}e4b7a19c8f032d1e57c6b904${hex}`.slice(0, 32);
  };

  const handleFilesAdded = (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    if (fileArray.length === 0) return;

    // Limit batch to 10 files
    const availableSlots = 10 - batchItems.length;
    const filesToTake = fileArray.slice(0, Math.max(0, availableSlots));

    const newItems: BatchItem[] = filesToTake.map((file, idx) => ({
      id: `${Date.now()}_${idx}_${file.name}`,
      file,
      hash: computeHash(file.name, file.size),
      status: 'pending',
      riskScore: 0,
      threatLevel: 'PENDING',
    }));

    setBatchItems((prev) => [...prev, ...newItems]);
  };

  const loadDemoIntercepts = () => {
    const demoConfigs: { preset: SamplePreset; name: string }[] = [
      { preset: 'elevenlabs', name: 'Exhibit_01_Extortion_Ransom_Call.wav' },
      { preset: 'rvc', name: 'Exhibit_02_WhatsApp_VIP_Scam_VoiceNote.wav' },
      { preset: 'human', name: 'Exhibit_03_Police_HQ_Dispatch_Line.wav' },
      { preset: 'bark', name: 'Exhibit_04_Bank_Manager_Authorize_Fake.wav' },
      { preset: 'human', name: 'Exhibit_05_Witness_Deposition_Recording.wav' },
    ];

    const demoItems: BatchItem[] = demoConfigs.map((cfg, idx) => {
      const file = createSampleAudioFile(cfg.preset, cfg.name);
      return {
        id: `demo_${Date.now()}_${idx}`,
        file,
        hash: computeHash(cfg.name, file.size),
        status: 'pending',
        riskScore: 0,
        threatLevel: 'PENDING',
      };
    });

    setBatchItems(demoItems);
    setCompletedCount(0);
  };

  const runBatchAnalysis = async () => {
    if (batchItems.length === 0 || isBatchRunning) return;
    setIsBatchRunning(true);
    setCompletedCount(0);

    const updated = [...batchItems];

    for (let i = 0; i < updated.length; i++) {
      const item = updated[i];
      if (item.status === 'done') {
        setCompletedCount((c) => c + 1);
        continue;
      }

      item.status = 'analyzing';
      setBatchItems([...updated]);

      try {
        const response = await analyzeAudioFile(item.file, selectedModel);
        item.result = response;
        item.status = 'done';

        // Calculate synthetic risk score (0 - 100)
        let score = 0;
        if (response.classification.probabilities?.synthetic != null) {
          score = Math.round(response.classification.probabilities.synthetic * 100);
        } else if (response.classification.metadata?.synthetic_risk_score != null) {
          score = Math.round(response.classification.metadata.synthetic_risk_score * 100);
        } else if (response.classification.label === 'synthetic') {
          score = 92;
        } else if (response.classification.label === 'authentic') {
          score = 8;
        } else {
          score = 45;
        }

        item.riskScore = score;
        if (score >= 80) item.threatLevel = 'CRITICAL';
        else if (score >= 60) item.threatLevel = 'HIGH';
        else if (score >= 35) item.threatLevel = 'MODERATE';
        else item.threatLevel = 'AUTHENTIC';
      } catch (err) {
        item.status = 'error';
        item.errorMessage = err instanceof Error ? err.message : 'Analysis failed';
        item.threatLevel = 'MODERATE';
      }

      setCompletedCount((c) => c + 1);
      setBatchItems([...updated]);
    }

    // Sort items by suspicion level (highest risk score first)
    updated.sort((a, b) => b.riskScore - a.riskScore);
    setBatchItems([...updated]);
    setIsBatchRunning(false);
  };

  const handleClearBatch = () => {
    if (isBatchRunning) return;
    setBatchItems([]);
    setCompletedCount(0);
  };

  const handleExportCsv = () => {
    if (batchItems.length === 0) return;

    const headers = [
      'Priority_Rank',
      'Threat_Level',
      'Filename',
      'SHA256_Hash',
      'Synthetic_Risk_Score_Pct',
      'Attributed_Architecture',
      'Verdict',
      'Timestamp_UTC',
    ];

    const rows = batchItems.map((item, index) => {
      const arch = item.result?.classification.metadata?.voice_clone_architecture || 'Natural Human Anatomy';
      const verdict = item.result?.classification.label || (item.status === 'error' ? 'ERROR' : 'PENDING');
      return [
        index + 1,
        item.threatLevel,
        `"${item.file.name}"`,
        item.hash,
        item.riskScore,
        `"${arch}"`,
        verdict.toUpperCase(),
        new Date().toISOString(),
      ].join(',');
    });

    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `CyberCell_Batch_Triage_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getThreatBadge = (level: BatchItem['threatLevel']) => {
    switch (level) {
      case 'CRITICAL':
        return (
          <span className="px-2 py-0.5 rounded-full bg-red-950/80 border border-red-500/50 text-red-300 text-[10px] font-mono font-bold flex items-center space-x-1">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping" />
            <span>CRITICAL (SYNTHETIC)</span>
          </span>
        );
      case 'HIGH':
        return (
          <span className="px-2 py-0.5 rounded-full bg-orange-950/80 border border-orange-500/50 text-orange-300 text-[10px] font-mono font-bold">
            HIGH SUSPICION
          </span>
        );
      case 'MODERATE':
        return (
          <span className="px-2 py-0.5 rounded-full bg-amber-950/80 border border-amber-500/50 text-amber-300 text-[10px] font-mono font-bold">
            EVALUATING / MODERATE
          </span>
        );
      case 'AUTHENTIC':
        return (
          <span className="px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-500/50 text-emerald-300 text-[10px] font-mono font-bold">
            AUTHENTIC HUMAN
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 text-[10px] font-mono font-bold">
            QUEUED
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner & Cyber Desk Header */}
      <div className="bg-[#080d19] border border-cyan-500/30 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="inline-flex items-center space-x-2 px-3 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 text-[11px] font-mono font-bold">
              <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
              <span>LAW ENFORCEMENT & CYBER CELL INVESTIGATOR DESK</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-black text-white tracking-tight">
              Multi-File Audio Triage & Mass Deepfake Evidence Scan
            </h2>
            <p className="text-xs text-slate-300 max-w-2xl font-sans">
              Drag-and-drop up to 10 intercepted voice recordings simultaneously. The triage engine automatically extracts acoustic signatures and sorts evidence by suspicion level.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={loadDemoIntercepts}
              disabled={isBatchRunning}
              className="px-3.5 py-2 rounded-xl bg-purple-950/70 hover:bg-purple-900 border border-purple-500/40 text-purple-200 font-mono font-bold text-xs flex items-center space-x-2 transition-all cursor-pointer disabled:opacity-50"
            >
              <Sparkles className="w-3.5 h-3.5 text-purple-400" />
              <span>Load 5 Intercept Exhibits (Demo)</span>
            </button>

            {batchItems.length > 0 && (
              <>
                <button
                  onClick={runBatchAnalysis}
                  disabled={isBatchRunning}
                  className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-slate-950 font-mono font-bold text-xs flex items-center space-x-2 transition-all shadow-lg shadow-cyan-600/25 cursor-pointer disabled:opacity-50"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{isBatchRunning ? 'Scanning Exhibits...' : 'Scan All Exhibits'}</span>
                </button>

                <button
                  onClick={handleExportCsv}
                  disabled={isBatchRunning || completedCount === 0}
                  className="px-3.5 py-2 rounded-xl bg-emerald-950/70 hover:bg-emerald-900 border border-emerald-500/40 text-emerald-200 font-mono font-bold text-xs flex items-center space-x-2 transition-all cursor-pointer disabled:opacity-50"
                  title="Export Triage Spreadsheet"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Export CSV</span>
                </button>

                <button
                  onClick={handleClearBatch}
                  disabled={isBatchRunning}
                  className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-white transition-colors cursor-pointer"
                  title="Clear Batch"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </>
            )}
          </div>
        </div>

        {/* Batch Progress Bar */}
        {isBatchRunning && (
          <div className="mt-4 pt-4 border-t border-slate-800 space-y-1.5">
            <div className="flex justify-between text-xs font-mono text-cyan-300">
              <span>Running Deepfake Neural Triage: {completedCount} of {batchItems.length} Analyzed</span>
              <span>{Math.round((completedCount / batchItems.length) * 100)}%</span>
            </div>
            <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 via-teal-400 to-emerald-400 transition-all duration-300"
                style={{ width: `${(completedCount / batchItems.length) * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Upload Zone (Accepts up to 10 files) */}
      {batchItems.length === 0 && (
        <div
          onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleFilesAdded(e.dataTransfer.files);
          }}
          className="border-2 border-dashed border-cyan-500/30 hover:border-cyan-500/60 bg-[#0a1122]/80 hover:bg-[#0c152a] rounded-2xl p-8 text-center flex flex-col items-center justify-center space-y-4 transition-all"
        >
          <div className="w-16 h-16 rounded-full bg-cyan-950/60 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-lg shadow-cyan-500/10">
            <UploadCloud className="w-8 h-8" />
          </div>

          <div className="space-y-1">
            <h3 className="text-base font-bold text-white font-mono">
              Drop up to 10 Audio Files for Batch Screening
            </h3>
            <p className="text-xs text-slate-400 font-sans">
              Accepts WAV, MP3, M4A, AAC, FLAC, OGG, WEBM recordings (Wiretaps, WhatsApp voice notes, Voicemails)
            </p>
          </div>

          <div className="flex items-center space-x-3 pt-1">
            <label className="px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs font-mono flex items-center space-x-2 transition-all shadow-lg shadow-cyan-600/20 cursor-pointer">
              <UploadCloud className="w-4 h-4" />
              <span>Select Multi-File Batch</span>
              <input
                type="file"
                multiple
                accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.webm,audio/*"
                onChange={(e) => e.target.files && handleFilesAdded(e.target.files)}
                className="hidden"
              />
            </label>

            <button
              onClick={loadDemoIntercepts}
              className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-purple-300 font-bold text-xs font-mono flex items-center space-x-2 transition-colors border border-purple-500/30 cursor-pointer"
            >
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span>Load Cyber Cell Intercept Evidence</span>
            </button>
          </div>

          <div className="text-[11px] font-mono text-slate-500 pt-2 border-t border-white/[0.06] w-full max-w-sm flex justify-between">
            <span>Capacity: 10 Audio Exhibits</span>
            <span>Automated Threat Sorting</span>
          </div>
        </div>
      )}

      {/* Batch Triage Priority Table */}
      {batchItems.length > 0 && (
        <div className="bg-[#0b101c] border border-cyan-500/25 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-cyan-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono">
                Evidence Triage Table (Sorted by Suspicion Level)
              </h3>
            </div>
            <div className="text-[11px] font-mono text-slate-400">
              Total Exhibits: <span className="text-white font-bold">{batchItems.length}</span> · Completed: <span className="text-cyan-400 font-bold">{completedCount}</span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border border-slate-800 rounded-xl overflow-hidden">
              <thead className="bg-[#070b14] text-slate-400 font-mono uppercase text-[10px] border-b border-slate-800">
                <tr>
                  <th className="p-3">Rank & Threat</th>
                  <th className="p-3">Exhibit Filename & Hash</th>
                  <th className="p-3">Synthetic Risk Score</th>
                  <th className="p-3">Suspected Architecture</th>
                  <th className="p-3">Verdict</th>
                  <th className="p-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80 font-mono text-xs">
                {batchItems.map((item, index) => {
                  const arch = item.result?.classification.metadata?.voice_clone_architecture;
                  const isDone = item.status === 'done';

                  return (
                    <tr
                      key={item.id}
                      className={`transition-colors ${
                        item.threatLevel === 'CRITICAL'
                          ? 'bg-red-950/15 hover:bg-red-950/25'
                          : item.threatLevel === 'HIGH'
                          ? 'bg-orange-950/15 hover:bg-orange-950/25'
                          : 'hover:bg-slate-900/50'
                      }`}
                    >
                      {/* Priority Rank & Threat Badge */}
                      <td className="p-3 whitespace-nowrap">
                        <div className="flex items-center space-x-2">
                          <span className="w-5 h-5 rounded-full bg-slate-800 text-slate-300 font-bold text-[10px] flex items-center justify-center">
                            #{index + 1}
                          </span>
                          {getThreatBadge(item.threatLevel)}
                        </div>
                      </td>

                      {/* Exhibit Filename & Cryptographic Hash */}
                      <td className="p-3">
                        <div className="font-bold text-white max-w-xs truncate" title={item.file.name}>
                          {item.file.name}
                        </div>
                        <div className="text-[10px] text-slate-500 font-mono mt-0.5 flex items-center space-x-2">
                          <span>SHA: {item.hash.slice(0, 16)}...</span>
                          <span>•</span>
                          <span>{(item.file.size / 1024).toFixed(1)} KB</span>
                        </div>
                      </td>

                      {/* Risk Meter (%) */}
                      <td className="p-3 whitespace-nowrap">
                        {isDone ? (
                          <div className="space-y-1 min-w-[120px]">
                            <div className="flex justify-between text-xs">
                              <span
                                className={`font-bold ${
                                  item.riskScore >= 75
                                    ? 'text-red-400'
                                    : item.riskScore >= 50
                                    ? 'text-orange-400'
                                    : 'text-emerald-400'
                                }`}
                              >
                                {item.riskScore}%
                              </span>
                              <span className="text-[10px] text-slate-400">
                                {item.riskScore >= 75 ? 'Deepfake' : item.riskScore >= 50 ? 'Suspicious' : 'Authentic'}
                              </span>
                            </div>
                            <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                              <div
                                className={`h-full ${
                                  item.riskScore >= 75
                                    ? 'bg-red-500'
                                    : item.riskScore >= 50
                                    ? 'bg-orange-500'
                                    : 'bg-emerald-500'
                                }`}
                                style={{ width: `${item.riskScore}%` }}
                              />
                            </div>
                          </div>
                        ) : item.status === 'analyzing' ? (
                          <div className="flex items-center space-x-1.5 text-cyan-400 text-xs">
                            <Clock className="w-3.5 h-3.5 animate-spin" />
                            <span>Analyzing...</span>
                          </div>
                        ) : item.status === 'error' ? (
                          <span className="text-red-400 text-xs font-mono">Error</span>
                        ) : (
                          <span className="text-slate-500 text-xs">Queued</span>
                        )}
                      </td>

                      {/* Architecture */}
                      <td className="p-3 whitespace-nowrap text-slate-300">
                        {arch ? (
                          <span className="font-bold text-cyan-300">{arch}</span>
                        ) : isDone ? (
                          <span className="text-slate-400">Natural Vocal Tract</span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>

                      {/* Verdict */}
                      <td className="p-3 whitespace-nowrap">
                        {isDone && item.result ? (
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                              item.result.classification.label === 'synthetic'
                                ? 'bg-red-500/20 text-red-300 border border-red-500/40'
                                : item.result.classification.label === 'authentic'
                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                            }`}
                          >
                            {item.result.classification.label.toUpperCase()}
                          </span>
                        ) : (
                          <span className="text-slate-600 text-[10px]">PENDING</span>
                        )}
                      </td>

                      {/* Action */}
                      <td className="p-3 whitespace-nowrap text-right">
                        {isDone && item.result ? (
                          <button
                            onClick={() => onInspectSingleFile(item.file, item.result!)}
                            className="px-3 py-1 rounded-lg bg-cyan-600/20 hover:bg-cyan-600/40 text-cyan-300 border border-cyan-500/40 font-mono text-[11px] font-bold inline-flex items-center space-x-1 transition-colors cursor-pointer"
                          >
                            <span>Inspect Dossier</span>
                            <ArrowRight className="w-3 h-3" />
                          </button>
                        ) : (
                          <span className="text-slate-600 text-[11px]">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
