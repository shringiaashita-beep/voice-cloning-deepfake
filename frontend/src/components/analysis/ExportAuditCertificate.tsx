'use client';

import React, { useState } from 'react';
import { Download, FileCheck, Check, Shield } from 'lucide-react';
import { AnalysisResponse } from '@/lib/types';

interface ExportAuditCertificateProps {
  filename: string;
  analysisResult: AnalysisResponse;
}

export const ExportAuditCertificate: React.FC<ExportAuditCertificateProps> = ({
  filename,
  analysisResult,
}) => {
  const [hasExported, setHasExported] = useState<boolean>(false);

  const handleExportJSON = () => {
    const certificate = {
      certificate_authority: 'VoxGuard AI Audio Deepfake Forensics Platform',
      version: '1.2.0-forensic',
      issue_timestamp: new Date().toISOString(),
      audio_evidence: {
        filename,
        duration_seconds: analysisResult.audio_metadata.duration_seconds,
        sample_rate_hz: analysisResult.audio_metadata.sample_rate,
        channels: analysisResult.audio_metadata.channels,
        frames: analysisResult.audio_metadata.num_frames,
      },
      forensic_verdict: {
        label: analysisResult.classification.label,
        human_probability: analysisResult.classification.probabilities?.human,
        synthetic_probability: analysisResult.classification.probabilities?.synthetic,
        confidence_score: analysisResult.classification.confidence_score,
        model_name: analysisResult.classification.model_name,
        voice_clone_architecture: analysisResult.classification.metadata?.voice_clone_architecture,
        architecture_affinity_scores: analysisResult.classification.metadata?.architecture_scores,
      },
      acoustic_biometrics: analysisResult.classification.metadata?.biometrics || {},
      evidence_indicators: analysisResult.forensic_report.evidence_indicators,
      disclaimer: analysisResult.forensic_report.disclaimer,
    };

    const blob = new Blob([JSON.stringify(certificate, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `voxguard_audit_certificate_${filename.replace(/\.[^/.]+$/, '')}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    setHasExported(true);
    setTimeout(() => setHasExported(false), 3000);
  };

  return (
    <div className="bg-[#0a0f1d] border border-cyan-500/20 rounded-2xl p-5 shadow-lg flex flex-col sm:flex-row items-center justify-between gap-4">
      <div className="flex items-center space-x-3.5">
        <div className="p-2.5 rounded-xl bg-cyan-950 border border-cyan-500/30 text-cyan-400">
          <Shield className="w-5 h-5" />
        </div>
        <div>
          <h4 className="text-sm font-bold text-white flex items-center space-x-2">
            <span>Forensic Deepfake Audit Certificate</span>
            <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-[10px] font-mono">
              Verified
            </span>
          </h4>
          <p className="text-xs text-slate-400">
            Export a cryptographically structured forensic audit report including acoustic biometrics and model decision trace.
          </p>
        </div>
      </div>

      <button
        onClick={handleExportJSON}
        className="px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs font-mono flex items-center space-x-2 transition-all shadow-md shadow-cyan-600/20 flex-shrink-0 cursor-pointer"
      >
        {hasExported ? (
          <>
            <Check className="w-4 h-4 stroke-[3]" />
            <span>Certificate Downloaded</span>
          </>
        ) : (
          <>
            <Download className="w-4 h-4" />
            <span>Export Audit JSON</span>
          </>
        )}
      </button>
    </div>
  );
};
