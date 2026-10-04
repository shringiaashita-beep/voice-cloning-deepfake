import React, { useState } from 'react';
import {
  AlertOctagon,
  RefreshCw,
  FilePlus,
  Terminal,
  Check,
  Copy,
  Wifi,
  WifiOff,
  ChevronDown,
  ChevronUp,
  Server,
  Activity,
  ShieldAlert,
  Zap,
} from 'lucide-react';
import { fetchApiHealth } from '@/lib/api';

interface ErrorCardProps {
  errorMessage: string;
  errorCode?: string;
  onRetry: () => void;
  onChooseNewFile: () => void;
  onUseFallback?: () => void;
}

export const ErrorCard: React.FC<ErrorCardProps> = ({
  errorMessage,
  errorCode = 'ANALYSIS_ERROR',
  onRetry,
  onChooseNewFile,
  onUseFallback,
}) => {
  const [showTroubleshooting, setShowTroubleshooting] = useState<boolean>(false);
  const [copiedCommand, setCopiedCommand] = useState<boolean>(false);
  const [copiedLog, setCopiedLog] = useState<boolean>(false);
  const [healthState, setHealthState] = useState<'idle' | 'checking' | 'online' | 'offline'>('idle');
  const [healthMessage, setHealthMessage] = useState<string>('');

  const isVercelHost =
    typeof window !== 'undefined' &&
    (window.location.hostname.includes('vercel') ||
      (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1'));

  const isConnectionError =
    errorCode === 'UPLOAD_FAILED' ||
    errorMessage.toLowerCase().includes('fetch') ||
    errorMessage.toLowerCase().includes('connect') ||
    errorMessage.toLowerCase().includes('network') ||
    errorMessage.toLowerCase().includes('failed to fetch');

  const isEmptyFileError = errorCode === 'EMPTY_FILE';
  const isFormatError = errorCode === 'UNSUPPORTED_FORMAT' || errorCode === 'DECODE_ERROR';
  const isSizeError = errorCode === 'FILE_TOO_LARGE';

  // Format intelligent user-friendly message
  let userFriendlyMessage = errorMessage;
  if (isConnectionError) {
    if (isVercelHost) {
      userFriendlyMessage =
        'VoxGuard is currently deployed on Vercel Cloud. The Python FastAPI backend server is offline or not running locally on port 8000. Click "Run Web DSP Fallback" below for instant in-browser forensic analysis!';
    } else {
      userFriendlyMessage =
        'VoxGuard could not connect to the forensic analysis backend server. Please verify that the Python FastAPI server is active on port 8000, or use Web DSP mode.';
    }
  } else if (isEmptyFileError) {
    userFriendlyMessage =
      'The audio stream contained zero frames or could not be decoded. Please verify your microphone or audio file active signal.';
  } else if (isFormatError) {
    userFriendlyMessage =
      'The provided audio file format is invalid or corrupted. Supported formats include WAV, MP3, FLAC, OGG, AAC, and M4A.';
  } else if (isSizeError) {
    userFriendlyMessage =
      'The uploaded audio file exceeds the maximum analysis threshold. Please select a shorter sample or compressed file.';
  }

  const launchCommand = 'uvicorn app.main:app --reload --port 8000';

  const handleCopyCommand = () => {
    navigator.clipboard.writeText(launchCommand);
    setCopiedCommand(true);
    setTimeout(() => setCopiedCommand(false), 2000);
  };

  const handleCopyDiagnosticLog = () => {
    const diagnosticLog = JSON.stringify(
      {
        timestamp: new Date().toISOString(),
        isVercelHost,
        errorCode,
        errorMessage,
        userFriendlyMessage,
        targetApi: 'http://localhost:8000/api/v1',
        userAgent: typeof window !== 'undefined' ? window.navigator.userAgent : 'N/A',
      },
      null,
      2
    );
    navigator.clipboard.writeText(diagnosticLog);
    setCopiedLog(true);
    setTimeout(() => setCopiedLog(false), 2000);
  };

  const handleTestHealth = async () => {
    setHealthState('checking');
    setHealthMessage('Testing connection to backend API server (port 8000)...');

    try {
      const health = await fetchApiHealth();
      if (health && health.status === 'ok') {
        setHealthState('online');
        setHealthMessage(`Server is online! (Version ${health.version || '1.0.0'})`);
      } else {
        setHealthState('offline');
        setHealthMessage('Backend server responded with error status.');
      }
    } catch {
      setHealthState('offline');
      setHealthMessage('Connection failed: Port 8000 is unreachable.');
    }
  };

  return (
    <div className="forensic-panel p-6 sm:p-8 max-w-3xl mx-auto my-8 border-l-4 border-l-red-500 bg-gradient-to-b from-red-950/30 via-slate-900/90 to-slate-950/90 rounded-2xl shadow-2xl shadow-red-950/30 border border-red-500/20 backdrop-blur-xl relative overflow-hidden transition-all duration-300">
      {/* Background Subtle Glowing Accent */}
      <div className="absolute -right-20 -top-20 w-60 h-60 bg-red-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-6 border-b border-white/[0.08]">
        <div className="flex items-center space-x-3">
          <div className="p-3 rounded-xl bg-red-950/80 border border-red-500/40 text-red-400 shadow-lg shadow-red-500/10 flex-shrink-0">
            <AlertOctagon className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-red-950/70 border border-red-500/30 text-red-300 text-[10px] font-mono tracking-wider uppercase">
                <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping inline-block" />
                <span>Analysis Interrupted</span>
              </span>
              {isVercelHost && (
                <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 text-[10px] font-mono font-bold">
                  <Zap className="w-3 h-3 text-cyan-400" />
                  <span>Vercel Cloud Mode</span>
                </span>
              )}
            </div>
            <h3 className="text-xl font-black text-white tracking-tight font-sans">
              Analysis Could Not Be Completed
            </h3>
          </div>
        </div>

        {/* Error Code Pill */}
        <div className="font-mono text-xs px-3 py-1.5 rounded-lg bg-red-950/60 border border-red-500/30 text-red-300 flex items-center space-x-1.5 self-start sm:self-auto">
          <ShieldAlert className="w-3.5 h-3.5 text-red-400" />
          <span>Error Code:</span>
          <span className="font-bold text-red-200">{errorCode}</span>
        </div>
      </div>

      {/* Main Error Body Description */}
      <div className="py-6 space-y-4 font-sans">
        <p className="text-sm text-slate-200 leading-relaxed bg-slate-900/60 p-4 rounded-xl border border-white/5">
          {userFriendlyMessage}
        </p>

        {/* Live Server Health Indicator Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-slate-950/80 border border-white/10 text-xs">
          <div className="flex items-center space-x-2.5">
            <Server className="w-4 h-4 text-cyan-400" />
            <span className="text-slate-400 font-mono">Backend Status:</span>
            {healthState === 'idle' && (
              <span className="text-slate-400 font-mono flex items-center space-x-1">
                <Activity className="w-3.5 h-3.5 text-slate-500" />
                <span>Not Verified</span>
              </span>
            )}
            {healthState === 'checking' && (
              <span className="text-cyan-400 font-mono flex items-center space-x-1 animate-pulse">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Testing Connection...</span>
              </span>
            )}
            {healthState === 'online' && (
              <span className="text-emerald-400 font-mono font-bold flex items-center space-x-1">
                <Wifi className="w-3.5 h-3.5" />
                <span>ONLINE (Port 8000 Ready)</span>
              </span>
            )}
            {healthState === 'offline' && (
              <span className="text-red-400 font-mono font-bold flex items-center space-x-1">
                <WifiOff className="w-3.5 h-3.5" />
                <span>OFFLINE (Unreachable)</span>
              </span>
            )}
          </div>

          <button
            onClick={handleTestHealth}
            disabled={healthState === 'checking'}
            className="px-3 py-1 rounded-lg bg-cyan-950/80 hover:bg-cyan-900 border border-cyan-500/30 text-cyan-300 font-mono text-[11px] font-semibold transition-all flex items-center space-x-1.5 cursor-pointer disabled:opacity-50"
          >
            <Activity className="w-3 h-3" />
            <span>Test Connection</span>
          </button>
        </div>

        {healthMessage && (
          <p className={`text-xs font-mono px-3 py-1.5 rounded-lg border ${
            healthState === 'online'
              ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'
              : 'bg-red-950/40 border-red-500/30 text-red-300'
          }`}>
            {healthMessage}
          </p>
        )}
      </div>

      {/* Troubleshooting Collapsible Accordion */}
      {isConnectionError && (
        <div className="mb-6 rounded-xl border border-white/10 bg-slate-900/50 overflow-hidden font-sans">
          <button
            onClick={() => setShowTroubleshooting(!showTroubleshooting)}
            className="w-full px-4 py-3 flex items-center justify-between text-left text-xs font-mono font-bold text-slate-300 hover:text-white bg-slate-800/40 hover:bg-slate-800/70 transition-colors"
          >
            <div className="flex items-center space-x-2">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <span>
                {isVercelHost
                  ? 'Vercel Deployment & Local Backend Setup Guide'
                  : 'How to start the VoxGuard API Server (Troubleshooting Guide)'}
              </span>
            </div>
            {showTroubleshooting ? (
              <ChevronUp className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            )}
          </button>

          {showTroubleshooting && (
            <div className="p-4 space-y-4 text-xs text-slate-300 border-t border-white/5 bg-slate-950/60">
              {/* Option 1: Vercel Cloud Mode */}
              <div className="p-3 rounded-xl bg-cyan-950/30 border border-cyan-500/30 space-y-1.5">
                <div className="flex items-center space-x-2 font-mono font-bold text-cyan-300">
                  <Zap className="w-4 h-4 text-cyan-400" />
                  <span>Option 1: Vercel In-Browser DSP Mode (No Setup Required)</span>
                </div>
                <p className="text-slate-300 leading-relaxed text-[11px]">
                  Click the glowing <strong className="text-cyan-300">&quot;Run Web DSP Fallback&quot;</strong> button below to analyze your audio immediately inside your web browser without running a local backend server.
                </p>
              </div>

              {/* Option 2: Local Python Backend */}
              <div className="space-y-2">
                <div className="flex items-center space-x-2 font-mono font-bold text-slate-200">
                  <Server className="w-4 h-4 text-emerald-400" />
                  <span>Option 2: Connect Local Python FastAPI Backend</span>
                </div>
                <p className="leading-relaxed text-[11px]">
                  If you want full neural network server execution on your local GPU/CPU:
                </p>
                <ol className="list-decimal list-inside space-y-1.5 text-slate-300 font-sans text-[11px]">
                  <li>Open terminal in the project root directory.</li>
                  <li>Execute launch command:</li>
                </ol>

                <div className="p-3 rounded-lg bg-slate-900 border border-slate-700/60 font-mono text-[11px] text-cyan-300 flex items-center justify-between">
                  <span>{launchCommand}</span>
                  <button
                    onClick={handleCopyCommand}
                    className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[10px] font-sans flex items-center space-x-1 transition-colors border border-white/10"
                  >
                    {copiedCommand ? (
                      <>
                        <Check className="w-3 h-3 text-emerald-400" />
                        <span className="text-emerald-400">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3 text-slate-400" />
                        <span>Copy Command</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Action Buttons Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-white/[0.08] font-sans text-xs">
        <button
          onClick={handleCopyDiagnosticLog}
          className="px-3.5 py-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-slate-200 font-medium transition-colors flex items-center space-x-1.5 border border-white/10"
        >
          {copiedLog ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400">Log Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5 text-slate-400" />
              <span>Copy Technical Log</span>
            </>
          )}
        </button>

        <div className="flex flex-wrap items-center gap-2.5">
          {onUseFallback && (
            <button
              onClick={onUseFallback}
              className={`px-4 py-2 rounded-xl font-semibold transition-all flex items-center space-x-1.5 cursor-pointer ${
                isVercelHost
                  ? 'bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 text-slate-950 font-bold shadow-lg shadow-cyan-500/25 animate-pulse'
                  : 'bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-500/40 shadow-md shadow-cyan-950/50'
              }`}
            >
              <Zap className={`w-4 h-4 ${isVercelHost ? 'text-slate-950 fill-current' : 'text-cyan-400'}`} />
              <span>{isVercelHost ? 'Run Web DSP Fallback (Vercel Mode)' : 'Run Web DSP Fallback'}</span>
            </button>
          )}

          <button
            onClick={onChooseNewFile}
            className="px-4 py-2 rounded-xl bg-white/[0.06] hover:bg-white/10 text-slate-300 font-medium transition-colors flex items-center space-x-1.5 border border-white/5"
          >
            <FilePlus className="w-4 h-4 text-slate-400" />
            <span>Choose Another File</span>
          </button>

          <button
            onClick={onRetry}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white font-bold flex items-center space-x-2 transition-all shadow-lg shadow-red-600/30 cursor-pointer active:scale-95"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Try Again</span>
          </button>
        </div>
      </div>
    </div>
  );
};

