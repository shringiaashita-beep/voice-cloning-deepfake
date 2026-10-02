import React from 'react';
import { AlertOctagon, RefreshCw, FilePlus } from 'lucide-react';

interface ErrorCardProps {
  errorMessage: string;
  errorCode?: string;
  onRetry: () => void;
  onChooseNewFile: () => void;
}

export const ErrorCard: React.FC<ErrorCardProps> = ({
  errorMessage,
  errorCode = 'ANALYSIS_ERROR',
  onRetry,
  onChooseNewFile,
}) => {
  let userFriendlyMessage = errorMessage;
  if (errorMessage.toLowerCase().includes('fetch')) {
    userFriendlyMessage = 'VoxGuard could not connect to the analysis backend service. Please verify that the API server is running on port 8000.';
  } else if (errorCode === 'EMPTY_FILE') {
    userFriendlyMessage = 'The audio stream contained zero frames or could not be decoded. Please ensure your microphone is capturing active audio signal.';
  }

  return (
    <div className="forensic-panel p-8 max-w-2xl mx-auto my-8 border-l-4 border-l-red-500 space-y-6">
      <div className="flex items-start space-x-4">
        <div className="p-3 rounded-xl bg-red-950/60 border border-red-500/30 text-red-400 flex-shrink-0">
          <AlertOctagon className="w-6 h-6" />
        </div>

        <div className="space-y-2 font-sans">
          <h3 className="text-lg font-bold text-white tracking-tight">
            Analysis could not be completed
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            {userFriendlyMessage}
          </p>
          <div className="pt-2 font-mono text-[11px] text-slate-400">
            <span>Error code: </span>
            <span className="text-red-400 font-semibold">{errorCode}</span>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-end gap-3 pt-4 border-t border-white/[0.08] font-sans text-xs">
        <button
          onClick={onChooseNewFile}
          className="px-4 py-2 rounded-xl bg-white/[0.06] hover:bg-white/10 text-slate-300 font-medium transition-colors flex items-center space-x-1.5"
        >
          <FilePlus className="w-4 h-4 text-slate-400" />
          <span>Choose Another File</span>
        </button>

        <button
          onClick={onRetry}
          className="px-5 py-2.5 rounded-xl bg-red-600 hover:bg-red-500 text-white font-semibold flex items-center space-x-1.5 transition-colors shadow-lg shadow-red-600/20"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Try Again</span>
        </button>
      </div>
    </div>
  );
};
