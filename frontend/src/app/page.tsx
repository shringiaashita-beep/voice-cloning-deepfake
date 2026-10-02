'use client';

import React, { useState } from 'react';
import { Shield, Sparkles, UploadCloud, Mic, Globe, Users, ShieldAlert } from 'lucide-react';
import { AudioUploader } from '@/components/upload/AudioUploader';
import { SampleAudioCards, SamplePreset } from '@/components/upload/SampleAudioCards';
import { LiveMicrophoneRecorder } from '@/components/upload/LiveMicrophoneRecorder';
import { AudioUrlIngestorCard } from '@/components/upload/AudioUrlIngestorCard';
import { SpeakerComparisonView } from '@/components/comparison/SpeakerComparisonView';
import { CyberCellBatchInvestigator } from '@/components/batch/CyberCellBatchInvestigator';
import { LoadingPipeline } from '@/components/analysis/LoadingPipeline';
import { AnalysisHeader } from '@/components/analysis/AnalysisHeader';
import { StatusCard } from '@/components/analysis/StatusCard';
import { ModelCard } from '@/components/analysis/ModelCard';
import { AudioMetadataCard } from '@/components/analysis/AudioMetadataCard';
import { AcousticObservations } from '@/components/analysis/AcousticObservations';
import { EvidenceIndicators } from '@/components/analysis/EvidenceIndicators';
import { WaveformViewer } from '@/components/analysis/WaveformViewer';
import { SpectrogramViewer } from '@/components/analysis/SpectrogramViewer';
import { ProcessingStats } from '@/components/analysis/ProcessingStats';
import { ProcessingPipeline } from '@/components/analysis/ProcessingPipeline';
import { DisclaimerCard } from '@/components/analysis/DisclaimerCard';
import { ErrorCard } from '@/components/common/ErrorCard';
import { DeepfakeRiskMeter } from '@/components/analysis/DeepfakeRiskMeter';
import { VoiceCloneFingerprint } from '@/components/analysis/VoiceCloneFingerprint';
import { ExportAuditCertificate } from '@/components/analysis/ExportAuditCertificate';
import { PrintablePdfDossier } from '@/components/analysis/PrintablePdfDossier';
import { ModelSelector } from '@/components/common/ModelSelector';
import { EducationalGuide } from '@/components/common/EducationalGuide';
import { TrainingReadinessCard } from '@/components/analysis/TrainingReadinessCard';
import { analyzeAudioFile, analyzeAudioUrl, ApiError } from '@/lib/api';
import { AnalysisResponse } from '@/lib/types';
import { createSampleAudioFile } from '@/lib/audioGenerator';

type IngestionMode = 'upload' | 'mic' | 'url' | 'compare' | 'batch';

export default function Home() {
  const [activeTab, setActiveTab] = useState<IngestionMode>('upload');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [currentFilename, setCurrentFilename] = useState<string>('');
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('ml_classifier');
  const [errorDetails, setErrorDetails] = useState<{ message: string; code?: string } | null>(null);

  const handleFileAnalysis = async (file: File) => {
    setIsLoading(true);
    setErrorDetails(null);
    setCurrentFilename(file.name);
    setCurrentFile(file);

    try {
      const response = await analyzeAudioFile(file, selectedModel);
      setAnalysisResult(response);
    } catch (err) {
      if (err instanceof ApiError) {
        setErrorDetails({ message: err.message, code: err.code });
      } else {
        setErrorDetails({
          message: err instanceof Error ? err.message : 'An unexpected analysis error occurred.',
          code: 'ANALYSIS_FAILED',
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleUrlAnalysis = async (url: string) => {
    setIsLoading(true);
    setErrorDetails(null);
    const parsedName = url.split('/').pop()?.split('?')[0] || 'remote_audio.wav';
    setCurrentFilename(parsedName);
    setCurrentFile(null);

    try {
      const response = await analyzeAudioUrl(url, selectedModel);
      setAnalysisResult(response);
    } catch (err) {
      if (err instanceof ApiError) {
        setErrorDetails({ message: err.message, code: err.code });
      } else {
        setErrorDetails({
          message: err instanceof Error ? err.message : 'An unexpected audio ingestion error occurred.',
          code: 'URL_INGESTION_FAILED',
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setErrorDetails(null);
    setCurrentFilename('');
    setCurrentFile(null);
  };

  const handleSampleSelect = (preset: SamplePreset, sampleName: string) => {
    const sampleFile = createSampleAudioFile(preset, sampleName);
    handleFileAnalysis(sampleFile);
  };

  return (
    <div className="main-container py-8 space-y-8">
      {/* 1. Hero & Ingestion Workspace (Before Single File Analysis) */}
      {!analysisResult && !isLoading && !errorDetails && (
        <div className="max-w-4xl mx-auto space-y-8 py-4">
          {/* Hero Heading Section */}
          <div className="text-center space-y-3 font-sans">
            <div className="inline-flex items-center space-x-2 px-3.5 py-1 rounded-full bg-cyan-950/70 border border-cyan-500/40 text-cyan-300 text-xs font-mono shadow-md shadow-cyan-500/10">
              <Shield className="w-3.5 h-3.5" />
              <span>VOICE CLONING & DEEPFAKE FORENSICS PLATFORM</span>
            </div>
            <h2 className="text-3xl sm:text-5xl font-black text-white tracking-tight leading-tight">
              Detect AI Voice Clones & Synthetic Deepfakes
            </h2>
            <p className="text-sm text-slate-300 max-w-2xl mx-auto leading-relaxed">
              Enterprise acoustic defense engineered to expose neural vocoders (ElevenLabs), voice conversions (RVC), and diffusion speech (Bark/XTTS) with explainable signal evidence.
            </p>
          </div>

          {/* Model Mode Selector */}
          <ModelSelector
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
            disabled={isLoading}
          />

          {/* Top-Level Ingestion Mode Tabs */}
          <div className="bg-[#0b1120] border border-cyan-500/20 rounded-2xl p-1.5 flex flex-wrap gap-1.5 shadow-lg">
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex-1 min-w-[130px] px-3.5 py-2.5 rounded-xl font-mono text-xs font-bold flex items-center justify-center space-x-2 transition-all cursor-pointer ${
                activeTab === 'upload'
                  ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/25'
                  : 'bg-slate-900/60 hover:bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <UploadCloud className="w-4 h-4" />
              <span>Upload File</span>
            </button>

            <button
              onClick={() => setActiveTab('mic')}
              className={`flex-1 min-w-[130px] px-3.5 py-2.5 rounded-xl font-mono text-xs font-bold flex items-center justify-center space-x-2 transition-all cursor-pointer ${
                activeTab === 'mic'
                  ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/25'
                  : 'bg-slate-900/60 hover:bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <Mic className="w-4 h-4" />
              <span>Live Mic Record</span>
            </button>

            <button
              onClick={() => setActiveTab('url')}
              className={`flex-1 min-w-[130px] px-3.5 py-2.5 rounded-xl font-mono text-xs font-bold flex items-center justify-center space-x-2 transition-all cursor-pointer ${
                activeTab === 'url'
                  ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/25'
                  : 'bg-slate-900/60 hover:bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <Globe className="w-4 h-4" />
              <span>Audio URL</span>
            </button>

            <button
              onClick={() => setActiveTab('compare')}
              className={`flex-1 min-w-[150px] px-3.5 py-2.5 rounded-xl font-mono text-xs font-bold flex items-center justify-center space-x-2 transition-all cursor-pointer ${
                activeTab === 'compare'
                  ? 'bg-gradient-to-r from-purple-500 to-indigo-500 text-white shadow-md shadow-purple-500/25'
                  : 'bg-slate-900/60 hover:bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <Users className="w-4 h-4" />
              <span>A/B Voice Match</span>
            </button>

            <button
              onClick={() => setActiveTab('batch')}
              className={`flex-1 min-w-[170px] px-3.5 py-2.5 rounded-xl font-mono text-xs font-bold flex items-center justify-center space-x-2 transition-all cursor-pointer ${
                activeTab === 'batch'
                  ? 'bg-gradient-to-r from-red-600 via-rose-600 to-amber-600 text-white shadow-md shadow-red-600/25'
                  : 'bg-slate-900/60 hover:bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <ShieldAlert className="w-4 h-4 text-amber-300" />
              <span>Cyber Cell Batch</span>
            </button>
          </div>

          {/* Active Workspace View */}
          {activeTab === 'upload' && (
            <div className="space-y-6">
              <AudioUploader onFileSelect={handleFileAnalysis} isLoading={isLoading} />
              <SampleAudioCards onSelectSample={handleSampleSelect} isLoading={isLoading} />
            </div>
          )}

          {activeTab === 'mic' && (
            <div className="space-y-6">
              <LiveMicrophoneRecorder onRecordingComplete={handleFileAnalysis} isLoading={isLoading} />
            </div>
          )}

          {activeTab === 'url' && (
            <div className="space-y-6">
              <AudioUrlIngestorCard onAnalyzeUrl={handleUrlAnalysis} isLoading={isLoading} />
            </div>
          )}

          {activeTab === 'compare' && (
            <div className="space-y-6">
              <SpeakerComparisonView selectedModel={selectedModel} />
            </div>
          )}

          {activeTab === 'batch' && (
            <div className="space-y-6">
              <CyberCellBatchInvestigator
                selectedModel={selectedModel}
                onInspectSingleFile={(file, result) => {
                  setCurrentFile(file);
                  setCurrentFilename(file.name);
                  setAnalysisResult(result);
                }}
              />
            </div>
          )}

          {/* Educational Guide Component */}
          <EducationalGuide />
        </div>
      )}

      {/* 2. Loading Pipeline State */}
      {isLoading && (
        <div className="py-8">
          <LoadingPipeline />
        </div>
      )}

      {/* 3. Error Card State */}
      {errorDetails && !isLoading && (
        <ErrorCard
          errorMessage={errorDetails.message}
          errorCode={errorDetails.code}
          onRetry={() => {
            if (currentFile) handleFileAnalysis(currentFile);
            else handleReset();
          }}
          onChooseNewFile={handleReset}
        />
      )}

      {/* 4. Results Dashboard (After Successful Single Audio Analysis) */}
      {analysisResult && !isLoading && (
        <div className="space-y-6 animate-fadeIn">
          {/* Header Bar */}
          <AnalysisHeader
            filename={currentFilename}
            analysisResult={analysisResult}
            onReset={handleReset}
          />

          {/* Court-Ready Printable PDF Dossier & Evidence Certificate */}
          <PrintablePdfDossier
            filename={currentFilename}
            analysisResult={analysisResult}
          />

          {/* Deepfake Risk Gauge & Probability Breakdown */}
          <DeepfakeRiskMeter classification={analysisResult.classification} />

          {/* Voice Cloning Architecture Fingerprint Card */}
          <VoiceCloneFingerprint classification={analysisResult.classification} />

          {/* Analysis Status & Verdict Card */}
          <StatusCard classification={analysisResult.classification} />

          {/* Export Audit Certificate (JSON) */}
          <ExportAuditCertificate
            filename={currentFilename}
            analysisResult={analysisResult}
          />

          {/* 2-Column Grid: Audio Profile & Processing Pipeline */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <AudioMetadataCard metadata={analysisResult.audio_metadata} />
            <ProcessingPipeline classification={analysisResult.classification} />
          </div>

          {/* Model Information Card */}
          <ModelCard classification={analysisResult.classification} />

          {/* Waveform Visualization */}
          <WaveformViewer
            waveform={analysisResult.visualization.waveform}
            audioFile={currentFile}
          />

          {/* Spectral Energy Distribution */}
          <SpectrogramViewer
            spectrogram={analysisResult.visualization.spectrogram}
            durationSeconds={analysisResult.audio_metadata.duration_seconds}
          />

          {/* Acoustic Observations */}
          <AcousticObservations observations={analysisResult.forensic_report.acoustic_observations} />

          {/* Evidence & Provenance Indicators */}
          <EvidenceIndicators
            indicators={analysisResult.forensic_report.evidence_indicators}
            isAnalysisOnly={analysisResult.classification.status === 'analysis_only'}
          />

          {/* Educational Guide & Learning Center */}
          <EducationalGuide />

          {/* Training Dataset Readiness Card */}
          <TrainingReadinessCard />

          {/* Technical Details & Processing Stats */}
          <ProcessingStats classification={analysisResult.classification} />

          {/* Mandatory Forensic Disclaimer Panel */}
          <DisclaimerCard disclaimerText={analysisResult.forensic_report.disclaimer} />

          {/* Bottom Reset Button */}
          <div className="pt-4 flex justify-center">
            <button
              onClick={handleReset}
              className="px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs font-sans transition-colors shadow-lg shadow-cyan-600/20 cursor-pointer"
            >
              Analyze Another Recording
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
