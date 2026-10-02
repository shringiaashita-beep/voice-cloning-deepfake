'use client';

import React, { useState } from 'react';
import { Globe, Link2, Sparkles, AlertCircle, ArrowRight } from 'lucide-react';

interface AudioUrlIngestorCardProps {
  onAnalyzeUrl: (url: string) => void;
  isLoading: boolean;
}

export const AudioUrlIngestorCard: React.FC<AudioUrlIngestorCardProps> = ({
  onAnalyzeUrl,
  isLoading,
}) => {
  const [url, setUrl] = useState<string>('');
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    const trimmed = url.trim();
    if (!trimmed) {
      setValidationError('Please enter a valid HTTP or HTTPS audio URL.');
      return;
    }

    try {
      const parsed = new URL(trimmed);
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        setValidationError('Only HTTP and HTTPS URLs are supported.');
        return;
      }
    } catch {
      setValidationError('The entered string is not a valid URL format.');
      return;
    }

    onAnalyzeUrl(trimmed);
  };

  const sampleUrls = [
    {
      title: 'Synthesized AI Voice Stream (.wav)',
      url: 'https://raw.githubusercontent.com/pytorch/audio/main/test/assets/steam-train-whistle-daniel_simon.wav',
    },
    {
      title: 'Speech Benchmark Audio (.wav)',
      url: 'https://raw.githubusercontent.com/librosa/librosa/main/tests/data/test16k.wav',
    },
  ];

  return (
    <div className="bg-[#0b101c] border border-cyan-500/25 rounded-2xl p-6 space-y-4 shadow-xl">
      <div className="border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <Globe className="w-4 h-4 text-cyan-400" />
          <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
            Audio URL Remote Ingestion
          </h4>
        </div>
        <p className="text-xs text-slate-400 mt-0.5 font-sans">
          Analyze audio hosted online, cloud storage links, podcast streams, or direct web media.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
              <Link2 className="w-4 h-4" />
            </div>
            <input
              type="url"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                setValidationError(null);
              }}
              placeholder="https://example.com/audio/interview_recording.mp3"
              disabled={isLoading}
              className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-100 placeholder-slate-500 text-xs font-mono focus:outline-none focus:border-cyan-500 transition-colors"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading || !url.trim()}
            className="px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs font-mono flex items-center justify-center space-x-2 transition-all shadow-lg shadow-cyan-600/20 cursor-pointer disabled:opacity-50"
          >
            <span>Fetch & Analyze</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {validationError && (
          <div className="p-3 rounded-xl bg-red-950/40 border border-red-500/30 text-red-300 text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{validationError}</span>
          </div>
        )}
      </form>

      {/* Preset Fast-Test URLs */}
      <div className="pt-2 border-t border-slate-800/80">
        <span className="text-[11px] font-mono text-slate-400 block mb-2">
          💡 Quick test public audio links:
        </span>
        <div className="flex flex-wrap gap-2">
          {sampleUrls.map((sample, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setUrl(sample.url);
                setValidationError(null);
                onAnalyzeUrl(sample.url);
              }}
              disabled={isLoading}
              className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-cyan-300 text-[11px] font-mono transition-colors text-left flex items-center space-x-1.5 cursor-pointer disabled:opacity-50"
            >
              <Sparkles className="w-3 h-3 text-cyan-400" />
              <span>{sample.title}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
