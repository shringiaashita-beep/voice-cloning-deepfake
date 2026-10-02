'use client';

import React, { useState } from 'react';
import { Users, Upload, ArrowLeftRight, ShieldAlert, ShieldCheck, AlertTriangle, Sparkles, FileAudio, Check } from 'lucide-react';
import { compareSpeakers, ApiError } from '@/lib/api';
import { SpeakerComparisonData, SpeakerComparisonResponse } from '@/lib/types';
import { createSampleAudioFile } from '@/lib/audioGenerator';

interface SpeakerComparisonViewProps {
  selectedModel: string;
}

export const SpeakerComparisonView: React.FC<SpeakerComparisonViewProps> = ({ selectedModel }) => {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [comparisonResult, setComparisonResult] = useState<SpeakerComparisonResponse | null>(null);

  const handleRunComparison = async () => {
    if (!fileA || !fileB) {
      setErrorMsg('Please supply both Sample A (Questioned) and Sample B (Reference) to execute comparison.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);

    try {
      const res = await compareSpeakers(fileA, fileB, selectedModel);
      setComparisonResult(res);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Comparison failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadPresetCase = (caseType: 'impersonation' | 'same_human') => {
    setErrorMsg(null);
    setComparisonResult(null);

    if (caseType === 'impersonation') {
      const sampleA = createSampleAudioFile('elevenlabs', 'questioned_voice_call.wav');
      const sampleB = createSampleAudioFile('human', 'authentic_ceo_interview.wav');
      setFileA(sampleA);
      setFileB(sampleB);
    } else {
      const sampleA = createSampleAudioFile('human', 'speaker_call_recording_1.wav');
      const sampleB = createSampleAudioFile('human', 'speaker_reference_sample_2.wav');
      setFileA(sampleA);
      setFileB(sampleB);
    }
  };

  const handleReset = () => {
    setFileA(null);
    setFileB(null);
    setComparisonResult(null);
    setErrorMsg(null);
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-[#0b101d] border border-cyan-500/25 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <div className="flex items-center space-x-2">
              <Users className="w-5 h-5 text-cyan-400" />
              <h3 className="text-sm font-bold text-white tracking-tight uppercase font-mono">
                Reference Voice Comparison (A/B Acoustic Similarity Analysis)
              </h3>
            </div>
            <p className="text-xs text-slate-300 mt-1 font-sans">
              Compare an unverified recording (Sample A) against an authentic speaker profile (Sample B) to detect targeted voice clone impersonation.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => loadPresetCase('impersonation')}
              disabled={isLoading}
              className="px-3 py-1.5 rounded-lg bg-red-950/70 border border-red-500/40 text-red-300 font-mono text-xs hover:bg-red-900/50 transition-colors flex items-center space-x-1.5"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Load Impersonation Case</span>
            </button>

            <button
              onClick={() => loadPresetCase('same_human')}
              disabled={isLoading}
              className="px-3 py-1.5 rounded-lg bg-emerald-950/70 border border-emerald-500/40 text-emerald-300 font-mono text-xs hover:bg-emerald-900/50 transition-colors flex items-center space-x-1.5"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Load Authentic Match</span>
            </button>
          </div>
        </div>

        {/* Dual Upload Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Sample A */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-cyan-400 uppercase">
                Sample A: Questioned Voice
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/30">
                Suspicious Recording
              </span>
            </div>

            {fileA ? (
              <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-700/80 flex items-center justify-between">
                <div className="flex items-center space-x-2.5 overflow-hidden">
                  <FileAudio className="w-5 h-5 text-cyan-400 flex-shrink-0" />
                  <span className="text-xs font-mono text-slate-200 truncate">{fileA.name}</span>
                </div>
                <button
                  onClick={() => setFileA(null)}
                  className="text-xs text-slate-400 hover:text-red-400 ml-2 font-mono"
                >
                  Change
                </button>
              </div>
            ) : (
              <label className="border-2 border-dashed border-slate-800 hover:border-cyan-500/50 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer transition-colors bg-slate-950/50">
                <Upload className="w-5 h-5 text-slate-500 mb-1" />
                <span className="text-xs text-slate-400 font-mono">Upload Questioned Voice (WAV, MP3, MP4, M4A, etc.)</span>
                <input
                  type="file"
                  accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.webm,.opus,.wma,.aiff,.aif,.caf,.amr,.3gp,.mp4,audio/*,video/webm,video/mp4"
                  onChange={(e) => e.target.files?.[0] && setFileA(e.target.files[0])}
                  className="hidden"
                />
              </label>
            )}
          </div>

          {/* Sample B */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-emerald-400 uppercase">
                Sample B: Reference Voice
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-500/30">
                Verified Authentic Speaker
              </span>
            </div>

            {fileB ? (
              <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-700/80 flex items-center justify-between">
                <div className="flex items-center space-x-2.5 overflow-hidden">
                  <FileAudio className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                  <span className="text-xs font-mono text-slate-200 truncate">{fileB.name}</span>
                </div>
                <button
                  onClick={() => setFileB(null)}
                  className="text-xs text-slate-400 hover:text-red-400 ml-2 font-mono"
                >
                  Change
                </button>
              </div>
            ) : (
              <label className="border-2 border-dashed border-slate-800 hover:border-emerald-500/50 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer transition-colors bg-slate-950/50">
                <Upload className="w-5 h-5 text-slate-500 mb-1" />
                <span className="text-xs text-slate-400 font-mono">Upload Reference Speaker Audio (WAV, MP3, MP4, etc.)</span>
                <input
                  type="file"
                  accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.webm,.opus,.wma,.aiff,.aif,.caf,.amr,.3gp,.mp4,audio/*,video/webm,video/mp4"
                  onChange={(e) => e.target.files?.[0] && setFileB(e.target.files[0])}
                  className="hidden"
                />
              </label>
            )}
          </div>
        </div>

        {errorMsg && (
          <div className="p-3 rounded-xl bg-red-950/40 border border-red-500/30 text-red-300 text-xs">
            {errorMsg}
          </div>
        )}

        {/* Submit Comparison Button */}
        <div className="flex justify-end pt-2">
          <button
            onClick={handleRunComparison}
            disabled={isLoading || !fileA || !fileB}
            className="px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs font-mono flex items-center space-x-2 transition-all shadow-lg shadow-cyan-600/20 cursor-pointer disabled:opacity-50"
          >
            <ArrowLeftRight className="w-4 h-4" />
            <span>{isLoading ? 'Running Acoustic Comparison...' : 'Run Forensic A/B Acoustic Comparison'}</span>
          </button>
        </div>
      </div>

      {/* Comparison Results Card */}
      {comparisonResult && (
        <div className="forensic-card rounded-2xl p-6 space-y-6 animate-fadeIn">
          {/* Verdict Banner */}
          <div className={`p-4 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
            comparisonResult.comparison.impersonation_risk === 'CRITICAL_IMPERSONATION'
              ? 'bg-red-950/70 border-red-500/40'
              : (comparisonResult.comparison.impersonation_risk === 'AUTHENTIC_MATCH'
                  ? 'bg-emerald-950/70 border-emerald-500/40'
                  : 'bg-cyan-950/70 border-cyan-500/40')
          }`}>
            <div className="flex items-center space-x-3.5">
              {comparisonResult.comparison.impersonation_risk === 'CRITICAL_IMPERSONATION' ? (
                <ShieldAlert className="w-7 h-7 text-red-400 animate-pulse flex-shrink-0" />
              ) : (
                <ShieldCheck className="w-7 h-7 text-emerald-400 flex-shrink-0" />
              )}
              <div>
                <span className="text-[10px] font-mono font-bold tracking-wider uppercase text-slate-400">
                  Forensic Comparison Finding
                </span>
                <h4 className="text-xl font-black text-white">
                  {comparisonResult.comparison.verdict}
                </h4>
              </div>
            </div>

            <div className="text-right font-mono bg-slate-950/80 px-4 py-2 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase block">Acoustic Similarity</span>
              <span className={`text-2xl font-black ${
                comparisonResult.comparison.speaker_similarity_percent >= 70 ? 'text-cyan-300' : 'text-slate-400'
              }`}>
                {comparisonResult.comparison.speaker_similarity_percent}%
              </span>
            </div>
          </div>

          {/* Side-by-Side Biometric Profile Matrix */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Sample A Summary */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 font-mono text-xs">
              <span className="text-slate-400 uppercase font-bold text-[11px] block border-b border-slate-800 pb-2">
                Sample A: Questioned Recording
              </span>
              <div className="space-y-1.5 text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-500">File:</span>
                  <span className="font-bold text-slate-200">{comparisonResult.comparison.sample_a.filename}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Verdict:</span>
                  <span className={`font-bold uppercase ${
                    comparisonResult.comparison.sample_a.label === 'synthetic' ? 'text-red-400' : 'text-emerald-400'
                  }`}>
                    {comparisonResult.comparison.sample_a.label}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Synthetic Probability:</span>
                  <span className="font-bold text-slate-200">
                    {comparisonResult.comparison.sample_a.synthetic_probability
                      ? `${(comparisonResult.comparison.sample_a.synthetic_probability * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            {/* Sample B Summary */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 font-mono text-xs">
              <span className="text-slate-400 uppercase font-bold text-[11px] block border-b border-slate-800 pb-2">
                Sample B: Reference Speaker
              </span>
              <div className="space-y-1.5 text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-500">File:</span>
                  <span className="font-bold text-slate-200">{comparisonResult.comparison.sample_b.filename}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Verdict:</span>
                  <span className={`font-bold uppercase ${
                    comparisonResult.comparison.sample_b.label === 'synthetic' ? 'text-red-400' : 'text-emerald-400'
                  }`}>
                    {comparisonResult.comparison.sample_b.label}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Synthetic Probability:</span>
                  <span className="font-bold text-slate-200">
                    {comparisonResult.comparison.sample_b.synthetic_probability
                      ? `${(comparisonResult.comparison.sample_b.synthetic_probability * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Investigative Findings List */}
          <div className="p-4 rounded-xl bg-[#090e1a] border border-slate-800 space-y-2">
            <span className="text-xs font-mono font-bold text-cyan-400 uppercase block">
              Forensic Investigation Findings:
            </span>
            <ul className="space-y-1.5 text-xs text-slate-300 font-sans">
              {comparisonResult.comparison.findings.map((f, i) => (
                <li key={i} className="flex items-start space-x-2">
                  <span className="text-cyan-400 mt-0.5">•</span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="flex justify-center pt-2">
            <button
              onClick={handleReset}
              className="px-6 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs font-mono transition-colors"
            >
              Clear Comparison Case
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
