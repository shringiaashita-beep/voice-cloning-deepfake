'use client';

import React from 'react';
import { ShieldAlert, ShieldCheck, AlertTriangle, Cpu, Radio, Activity } from 'lucide-react';
import { Classification } from '@/lib/types';

interface DeepfakeRiskMeterProps {
  classification: Classification;
}

export const DeepfakeRiskMeter: React.FC<DeepfakeRiskMeterProps> = ({ classification }) => {
  const isReady = classification.status === 'ready';
  const synthProb = classification.probabilities?.synthetic ?? null;
  const humanProb = classification.probabilities?.human ?? null;
  const confidence = classification.confidence_score ?? null;

  // Percentage from 0 to 100 (synthetic + human = 100%)
  const riskPercent = synthProb !== null ? Math.round(synthProb * 1000) / 10 : null;
  const humanPercent = humanProb !== null ? Math.round(humanProb * 1000) / 10 : null;

  const biometrics = classification.metadata?.biometrics;
  const arch = classification.metadata?.voice_clone_architecture || 'Neural Audio Classifier';

  // Determine risk level and theme from canonical classification label
  let levelColor = 'text-cyan-400';
  let levelBg = 'bg-cyan-950/60 border-cyan-500/30';
  let levelTitle = 'ANALYSIS ONLY';
  let headlineText = 'Acoustic Baseline Observation';
  let strokeColor = '#06b6d4';
  let badgeIcon = <Cpu className="w-4 h-4 text-cyan-400" />;

  if (isReady) {
    if (classification.label === 'synthetic') {
      levelColor = 'text-red-400';
      levelBg = 'bg-red-950/70 border-red-500/40';
      levelTitle = 'SYNTHETIC SPEECH DETECTED';
      headlineText = 'Synthetic Speech Assessment';
      strokeColor = '#ef4444';
      badgeIcon = <ShieldAlert className="w-4 h-4 text-red-400 animate-pulse" />;
    } else if (classification.label === 'human') {
      levelColor = 'text-emerald-400';
      levelBg = 'bg-emerald-950/70 border-emerald-500/40';
      levelTitle = 'GENUINE HUMAN SPEECH';
      headlineText = 'Human Speech Assessment';
      strokeColor = '#10b981';
      badgeIcon = <ShieldCheck className="w-4 h-4 text-emerald-400" />;
    } else {
      levelColor = 'text-cyan-400';
      levelBg = 'bg-cyan-950/70 border-cyan-500/40';
      levelTitle = 'BORDERLINE / INCONCLUSIVE';
      headlineText = 'Analysis Inconclusive';
      strokeColor = '#06b6d4';
      badgeIcon = <Radio className="w-4 h-4 text-cyan-400" />;
    }
  }

  // Calculate SVG circular gauge
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = riskPercent !== null
    ? circumference - (riskPercent / 100) * circumference
    : circumference * 0.5;

  return (
    <div className="forensic-card rounded-2xl p-6 relative overflow-hidden">
      {/* Background glow overlay */}
      <div 
        className="absolute -top-16 -right-16 w-56 h-56 rounded-full blur-3xl opacity-20 pointer-events-none transition-colors duration-500"
        style={{ backgroundColor: strokeColor }}
      />

      <div className="flex flex-col lg:flex-row items-center justify-between gap-6 relative z-10">
        {/* Left: Gauge Display */}
        <div className="flex items-center space-x-6">
          <div className="relative w-36 h-36 flex items-center justify-center">
            <svg className="w-36 h-36 transform -rotate-90">
              {/* Background track */}
              <circle
                cx="72"
                cy="72"
                r={radius}
                stroke="#1e293b"
                strokeWidth="12"
                fill="transparent"
              />
              {/* Animated Progress circle */}
              <circle
                cx="72"
                cy="72"
                r={radius}
                stroke={strokeColor}
                strokeWidth="12"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                fill="transparent"
                className="transition-all duration-1000 ease-out"
              />
            </svg>

            {/* Center percentage label */}
            <div className="absolute flex flex-col items-center justify-center text-center">
              <span className={`text-2xl font-black font-mono tracking-tight ${levelColor}`}>
                {riskPercent !== null ? `${riskPercent}%` : 'N/A'}
              </span>
              <span className="text-[10px] uppercase font-mono text-slate-400 tracking-wider">
                Synthetic Risk
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <div className={`inline-flex items-center space-x-2 px-3 py-1 rounded-full border text-xs font-mono font-bold uppercase ${levelBg}`}>
                {badgeIcon}
                <span>{levelTitle}</span>
              </div>

              {humanPercent !== null && (
                <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 text-xs font-mono font-bold">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                  <span>HUMAN PROBABILITY: {humanPercent}%</span>
                </div>
              )}
            </div>

            <h3 className="text-xl font-extrabold text-white">
              {headlineText}
            </h3>

            <p className="text-xs text-slate-300 max-w-md leading-relaxed font-sans">
              Classification: <span className="font-bold uppercase text-white">{classification.label}</span>. Model: <span className="font-semibold text-cyan-300">{classification.model_name}</span>. Synthetic Probability: <span className="font-bold text-red-400">{riskPercent !== null ? `${riskPercent}%` : 'N/A'}</span> | Human Probability: <span className="font-bold text-emerald-400">{humanPercent !== null ? `${humanPercent}%` : 'N/A'}</span> (Model Confidence: {confidence !== null ? `${Math.round(confidence * 100)}%` : 'N/A'}).
            </p>
          </div>
        </div>

        {/* Right: Dual Bar Breakdown & Biometrics */}
        <div className="w-full lg:w-72 bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-3 font-mono">
          <div className="flex justify-between items-center text-xs text-slate-400 pb-2 border-b border-slate-800">
            <span className="flex items-center space-x-1.5">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span>Probability Split</span>
            </span>
            <span className="text-[11px] text-cyan-400 font-semibold">{isReady ? 'Model Output' : 'Uncalibrated'}</span>
          </div>

          {/* AI Synthetic Bar */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300">Synthetic Probability:</span>
              <span className="text-red-400 font-bold">{riskPercent !== null ? `${riskPercent}%` : 'N/A'}</span>
            </div>
            <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-red-600 to-red-400 rounded-full transition-all duration-700"
                style={{ width: `${riskPercent || 0}%` }}
              />
            </div>
          </div>

          {/* Genuine Human Bar */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300">Human Probability:</span>
              <span className="text-emerald-400 font-bold">{humanPercent !== null ? `${humanPercent}%` : 'N/A'}</span>
            </div>
            <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full transition-all duration-700"
                style={{ width: `${humanPercent || 0}%` }}
              />
            </div>
          </div>

          {/* Quick Biometrics Footer */}
          {biometrics && (
            <div className="pt-2 border-t border-slate-800/80 grid grid-cols-2 gap-2 text-[10px] text-slate-400">
              <div>
                <span className="text-slate-500">Jitter:</span>{' '}
                <span className="text-slate-200 font-bold">
                  {biometrics.micro_jitter_index ? `${(biometrics.micro_jitter_index * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Vocoder Cut:</span>{' '}
                <span className="text-slate-200 font-bold">
                  {biometrics.vocoder_cutoff_score ? `${(biometrics.vocoder_cutoff_score * 100).toFixed(0)}%` : 'N/A'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
