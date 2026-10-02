/**
 * VoxGuard Frontend TypeScript Interfaces.
 * Strictly aligned with backend FastAPI schemas and forensic models.
 */

export type ProvenanceType = 'metadata' | 'signal_analysis' | 'heuristic' | 'ml_model';
export type SeverityType = 'LOW' | 'MEDIUM' | 'HIGH' | 'INFO';

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface AudioMetadata {
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  num_frames: number;
  metadata?: {
    original_sample_rate?: number;
    original_channels?: number;
    resampled?: boolean;
    channel_converted?: boolean;
    peak_scaling_factor?: number;
    dtype?: string;
    format?: string;
  };
}

export interface ClassificationProbabilities {
  human: number | null;
  synthetic: number | null;
}

export interface ArchitectureScores {
  elevenlabs_neural_vocoder?: number;
  rvc_voice_conversion?: number;
  bark_diffusion_tts?: number;
  natural_human_vocal_tract?: number;
}

export interface BiometricMetrics {
  high_mid_energy_ratio?: number;
  vocoder_cutoff_score?: number;
  micro_jitter_index?: number;
  high_order_mfcc_variance?: number;
  spectral_flux_mean?: number;
  spectral_flux_std?: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  status: string;
  is_active: boolean;
}

export interface Classification {
  label: string;
  probabilities: ClassificationProbabilities;
  confidence_score: number | null;
  model_name: string;
  model_version: string;
  status: 'ready' | 'analysis_only' | 'not_ready';
  metadata: {
    execution_time_ms?: number;
    device?: string;
    disclaimer?: string;
    evaluation_notice?: string;
    voice_clone_architecture?: string;
    architecture_scores?: ArchitectureScores;
    synthetic_risk_level?: string;
    synthetic_risk_score?: number;
    biometrics?: BiometricMetrics;
    acoustic_statistics?: {
      spectral_centroid_mean_hz?: number;
      spectral_centroid_std_hz?: number;
      spectral_centroid_min_hz?: number;
      spectral_centroid_max_hz?: number;
      spectral_rolloff_mean_hz?: number;
      spectral_rolloff_std_hz?: number;
      log_mel_mean_db?: number;
      log_mel_std_db?: number;
      high_band_mel_mean_db?: number;
      low_band_mel_mean_db?: number;
      mfcc_mean?: number;
      mfcc_std?: number;
      mfcc_variance?: number;
      temporal_variation_centroid_delta_mean?: number;
    };
  };
}

export interface WaveformPayload {
  target_points: number;
  min_peaks: number[];
  max_peaks: number[];
  peak_envelope: number[];
  duration_seconds: number;
}

export interface SpectrogramPayload {
  freq_bins: number;
  time_bins: number;
  data_grid: number[][];
  min_db: number;
  max_db: number;
}

export interface VisualizationPayload {
  waveform: WaveformPayload;
  spectrogram: SpectrogramPayload;
}

export interface EvidenceIndicator {
  name: string;
  measured_value: string;
  interpretation: string;
  severity: SeverityType;
  confidence_strength: string;
  is_model_derived: boolean;
  provenance: ProvenanceType;
}

export interface ForensicReportPayload {
  audio_metadata: {
    duration_seconds: number;
    sample_rate_hz: number;
    channels: number;
    num_frames: number;
    format: string;
  };
  acoustic_observations: {
    spectral_centroid?: {
      mean_hz: number;
      std_hz: number;
      min_hz: number;
      max_hz: number;
    };
    spectral_rolloff_85_percent?: {
      mean_hz: number;
      std_hz: number;
    };
    mfcc_statistics?: {
      num_coefficients: number;
      mean: number;
      variance: number;
    };
    signal_characteristics?: {
      average_energy_db: number;
      dynamic_range_db: number;
    };
    note?: string;
  };
  model_assessment: {
    verdict: string;
    probabilities: ClassificationProbabilities;
    confidence_score: number | null;
    model_name: string;
    model_version: string;
  } | null;
  evidence_indicators: EvidenceIndicator[];
  disclaimer: string;
  timestamp: string;
}

export interface AnalysisResponse {
  success: boolean;
  audio_metadata: AudioMetadata;
  classification: Classification;
  visualization: VisualizationPayload;
  forensic_report: ForensicReportPayload;
}

export interface ApiErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
  };
}

export interface TrainingStatusResponse {
  dataset_available: boolean;
  feature_artifacts: number;
  windows: number;
  speakers: number;
  class_balance: {
    human: number;
    synthetic: number;
  };
  speaker_leakage: boolean;
  feature_integrity: boolean;
  ood_available: boolean;
  training_ready: boolean;
  validation_errors?: string[];
}

export interface ComparisonMetrics {
  mfcc_cosine_similarity: number;
  pitch_alignment_score: number;
  spectral_rolloff_similarity: number;
  centroid_delta_hz: number;
}

export interface ComparisonSampleSummary {
  filename: string;
  label: string;
  confidence_score: number | null;
  synthetic_probability: number | null;
  model_name: string;
  duration_seconds: number;
}

export interface SpeakerComparisonData {
  speaker_similarity_percent: number;
  impersonation_risk: string;
  verdict: string;
  findings: string[];
  sample_a: ComparisonSampleSummary;
  sample_b: ComparisonSampleSummary;
  metrics: ComparisonMetrics;
}

export interface SpeakerComparisonResponse {
  success: boolean;
  comparison: SpeakerComparisonData;
  sample_a_analysis: AnalysisResponse;
  sample_b_analysis: AnalysisResponse;
}

