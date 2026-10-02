/**
 * VoxGuard Frontend API Client Service.
 * Communicates with FastAPI backend endpoints.
 */

import { AnalysisResponse, ApiErrorResponse, HealthResponse, ModelInfo, TrainingStatusResponse, SpeakerComparisonResponse } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export class ApiError extends Error {
  code: string;
  statusCode: number;

  constructor(message: string, code: string = 'API_ERROR', statusCode: number = 500) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.statusCode = statusCode;
  }
}

/**
 * Checks API server health status.
 */
export async function fetchApiHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      throw new Error(`Health check failed with HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : 'Unable to connect to VoxGuard API server.',
      'NETWORK_ERROR',
      503
    );
  }
}

/**
 * Analyzes remote audio directly from a URL.
 */
export async function analyzeAudioUrl(url: string, modelId: string = 'voice_clone_detector'): Promise<AnalysisResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze/url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ url, model: modelId }),
    });

    const data = await response.json();
    if (!response.ok || data.success === false) {
      const errorData = data as ApiErrorResponse;
      const errorCode = errorData?.error?.code || `HTTP_${response.status}`;
      const errorMessage = errorData?.error?.message || `URL Analysis failed with HTTP ${response.status}`;
      throw new ApiError(errorMessage, errorCode, response.status);
    }

    return data as AnalysisResponse;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(
      error instanceof Error ? error.message : 'Failed to fetch and analyze audio URL.',
      'URL_ANALYSIS_FAILED',
      500
    );
  }
}

/**
 * Compares two speaker samples for identity match and voice clone impersonation.
 */
export async function compareSpeakers(
  sampleA: File,
  sampleB: File,
  modelId: string = 'voice_clone_detector'
): Promise<SpeakerComparisonResponse> {
  const formData = new FormData();
  formData.append('sample_a', sampleA);
  formData.append('sample_b', sampleB);

  const endpointUrl = `${API_BASE_URL}/compare?model=${encodeURIComponent(modelId)}`;

  try {
    const response = await fetch(endpointUrl, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    const data = await response.json();
    if (!response.ok || data.success === false) {
      const errorData = data as ApiErrorResponse;
      const errorCode = errorData?.error?.code || `HTTP_${response.status}`;
      const errorMessage = errorData?.error?.message || `Comparison failed with HTTP ${response.status}`;
      throw new ApiError(errorMessage, errorCode, response.status);
    }

    return data as SpeakerComparisonResponse;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(
      error instanceof Error ? error.message : 'An error occurred during voice comparison.',
      'COMPARISON_FAILED',
      500
    );
  }
}


/**
 * Uploads an audio file for forensic analysis.
 * Defaults to 'voice_clone_detector' for active voice cloning deepfake classification.
 */
export async function analyzeAudioFile(file: File, modelId: string = 'voice_clone_detector'): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const url = modelId
    ? `${API_BASE_URL}/analyze?model=${encodeURIComponent(modelId)}`
    : `${API_BASE_URL}/analyze`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok || data.success === false) {
      const errorData = data as ApiErrorResponse;
      const errorCode = errorData?.error?.code || `HTTP_${response.status}`;
      const errorMessage = errorData?.error?.message || `Analysis failed with HTTP ${response.status}`;
      throw new ApiError(errorMessage, errorCode, response.status);
    }

    return data as AnalysisResponse;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(
      error instanceof Error ? error.message : 'An error occurred during audio upload.',
      'UPLOAD_FAILED',
      500
    );
  }
}

/**
 * Fetches the list of registered detection models from the backend.
 */
export async function fetchModels(): Promise<ModelInfo[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/models`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      cache: 'no-store',
    });
    if (!response.ok) return [];
    const data = await response.json();
    return data?.models || [];
  } catch {
    return [];
  }
}

/**
 * Selects the globally active classifier model on the backend.
 */
export async function selectModel(modelId: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/models/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Fetches training dataset infrastructure status.
 */
export async function fetchTrainingStatus(): Promise<TrainingStatusResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/training/status`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      throw new Error(`Training status check failed with HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : 'Unable to fetch training dataset status.',
      'TRAINING_STATUS_FAILED',
      503
    );
  }
}
