import React from 'react';
import { Shield } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="w-full border-t border-white/[0.08] bg-[#070a10] py-6 mt-16 font-sans text-xs text-slate-400">
      <div className="max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center space-x-2">
          <Shield className="w-4 h-4 text-cyan-400/80" />
          <span>VoxGuard AI Audio Forensics Workstation</span>
        </div>
        <div className="flex items-center space-x-6 font-mono text-[11px] text-slate-500">
          <span>Signal Analysis Mode</span>
          <span>Max: 10 MB</span>
          <span>WAV · MP3 · FLAC · OGG</span>
        </div>
      </div>
    </footer>
  );
};
