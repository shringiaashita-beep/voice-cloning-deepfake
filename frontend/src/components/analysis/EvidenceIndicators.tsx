'use client';

import React, { useState } from 'react';
import { Tag, Filter, Check } from 'lucide-react';
import { EvidenceIndicator, ProvenanceType } from '@/lib/types';

interface EvidenceIndicatorsProps {
  indicators: EvidenceIndicator[];
  isAnalysisOnly?: boolean;
}

export const EvidenceIndicators: React.FC<EvidenceIndicatorsProps> = ({
  indicators,
  isAnalysisOnly = true
}) => {
  const [filter, setFilter] = useState<string>('ALL');
  const [expandedIndices, setExpandedIndices] = useState<Set<number>>(new Set());

  const toggleExpand = (idx: number) => {
    setExpandedIndices((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  const availableFilters = isAnalysisOnly
    ? ['ALL', 'METADATA', 'SIGNAL_ANALYSIS', 'HEURISTIC']
    : ['ALL', 'METADATA', 'SIGNAL_ANALYSIS', 'HEURISTIC', 'ML_MODEL'];

  const filteredIndicators = indicators.filter((ind) => {
    if (filter === 'ALL') return true;
    return ind.provenance.toUpperCase() === filter;
  });

  const getProvenanceBadge = (provenance: ProvenanceType) => {
    switch (provenance) {
      case 'metadata':
        return (
          <span className="px-2 py-0.5 rounded bg-blue-950/80 border border-blue-500/40 text-blue-300 font-mono text-[10px] uppercase font-semibold">
            PROVENANCE: METADATA
          </span>
        );
      case 'signal_analysis':
        return (
          <span className="px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 font-mono text-[10px] uppercase font-semibold">
            PROVENANCE: SIGNAL ANALYSIS
          </span>
        );
      case 'heuristic':
        return (
          <span className="px-2 py-0.5 rounded bg-amber-950/80 border border-amber-500/40 text-amber-300 font-mono text-[10px] uppercase font-semibold">
            PROVENANCE: HEURISTIC
          </span>
        );
      case 'ml_model':
        return (
          <span className="px-2 py-0.5 rounded bg-purple-950/80 border border-purple-500/40 text-purple-300 font-mono text-[10px] uppercase font-semibold">
            PROVENANCE: ML MODEL
          </span>
        );
      default:
        return null;
    }
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity.toUpperCase()) {
      case 'HIGH':
        return <span className="text-red-400 font-mono text-xs font-semibold">SEVERITY: HIGH</span>;
      case 'MEDIUM':
        return <span className="text-amber-400 font-mono text-xs font-semibold">SEVERITY: MEDIUM</span>;
      case 'LOW':
        return <span className="text-slate-400 font-mono text-xs font-semibold">SEVERITY: LOW</span>;
      default:
        return <span className="text-cyan-400 font-mono text-xs font-semibold">INFO</span>;
    }
  };

  return (
    <div className="forensic-card rounded-xl p-6 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold font-mono uppercase tracking-wider text-slate-200 flex items-center space-x-2">
            <Tag className="w-4 h-4 text-cyan-400" />
            <span>Forensic Evidence & Audio Clues</span>
          </h3>
          <p className="text-[11px] text-slate-400 font-sans mt-0.5">
            💡 Individual acoustic tests showing file health, audio clarity, and frequency signal checks.
          </p>
        </div>

        {/* Provenance Filter Tabs */}
        <div className="flex items-center space-x-1 font-mono text-xs bg-slate-950 p-1 rounded-lg border border-slate-800">
          {availableFilters.map((tag) => (
            <button
              key={tag}
              onClick={() => setFilter(tag)}
              className={`px-2.5 py-1 rounded transition-colors text-[11px] ${
                filter === tag
                  ? 'bg-cyan-950 text-cyan-300 border border-cyan-500/40 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tag.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3 font-mono">
        {filteredIndicators.map((ind, idx) => {
          const isExpanded = expandedIndices.has(idx);

          return (
            <div
              key={idx}
              className="bg-slate-950/70 p-4 rounded-lg border border-slate-800/80 space-y-3 transition-colors hover:border-slate-700"
            >
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                <div className="space-y-1.5 flex-1">
                  <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                    <h4 className="text-sm font-bold text-slate-100">{ind.name}</h4>
                    {getProvenanceBadge(ind.provenance)}
                    {ind.is_model_derived ? (
                      <span className="px-1.5 py-0.5 rounded bg-purple-950/50 text-purple-300 text-[9px] font-semibold border border-purple-500/30">
                        MODEL DERIVED
                      </span>
                    ) : (
                      <span className="px-1.5 py-0.5 rounded bg-slate-900 text-slate-400 text-[9px] font-semibold border border-slate-700">
                        OBSERVATIONAL
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    {ind.interpretation}
                  </p>
                  <div className="text-xs text-slate-400">
                    <span>Measured Value: </span>
                    <span className="text-cyan-300 font-semibold">{ind.measured_value}</span>
                  </div>
                </div>

                <div className="flex flex-col items-end space-y-2 font-mono">
                  {getSeverityBadge(ind.severity)}
                  <button
                    onClick={() => toggleExpand(idx)}
                    className="text-[10px] text-cyan-400 hover:text-cyan-300 underline"
                  >
                    {isExpanded ? 'Hide Details ▲' : 'Expand Details ▼'}
                  </button>
                </div>
              </div>

              {/* Expandable Details Box */}
              {isExpanded && (
                <div className="pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 space-y-1 bg-[#070b12] p-3 rounded-md">
                  <div>
                    <span className="text-slate-400">Provenance Axis: </span>
                    <span className="text-slate-200">{ind.provenance}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Confidence Strength: </span>
                    <span className="text-slate-200">{ind.confidence_strength || 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Model Derived Status: </span>
                    <span className="text-slate-200">
                      {ind.is_model_derived ? 'Yes (Evaluated Classifier Score)' : 'No (Objective Signal Measurement)'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {filteredIndicators.length === 0 && (
          <div className="text-center py-6 text-slate-400 text-xs font-mono">
            No indicators found for filter "{filter}".
          </div>
        )}
      </div>
    </div>
  );
};
