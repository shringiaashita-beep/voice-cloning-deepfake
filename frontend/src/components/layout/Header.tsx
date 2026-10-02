'use client';

import React, { useEffect, useState } from 'react';
import { Shield, Activity } from 'lucide-react';
import { fetchApiHealth } from '@/lib/api';

export const Header: React.FC = () => {
  const [isOnline, setIsOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await fetchApiHealth();
        setIsOnline(true);
      } catch (err) {
        setIsOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full h-[72px] bg-[#090d14]/90 backdrop-blur-md border-b border-white/[0.08]">
      <div className="max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 h-full flex items-center justify-between">
        {/* Left Branding */}
        <div className="flex items-center space-x-3.5">
          <div className="p-2 rounded-xl bg-cyan-950/50 border border-cyan-500/30 text-cyan-400">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2.5">
              <h1 className="text-lg font-bold tracking-tight text-white">
                VoxGuard
              </h1>
              <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-white/[0.06] border border-white/10 text-cyan-300">
                0.1.0-MVP
              </span>
            </div>
            <p className="text-xs text-slate-400 tracking-normal font-sans">
              Voice Forensics & Synthetic Speech Analysis
            </p>
          </div>
        </div>

        {/* Right API Status Pill */}
        <div className="flex items-center">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-white/10 text-xs font-mono">
            {isOnline === null ? (
              <>
                <Activity className="w-3.5 h-3.5 text-slate-500 animate-spin" />
                <span className="text-slate-400">CONNECTING...</span>
              </>
            ) : isOnline ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-400"></span>
                </span>
                <span className="text-cyan-400 font-semibold">API ONLINE</span>
              </>
            ) : (
              <>
                <span className="h-2 w-2 rounded-full bg-red-500"></span>
                <span className="text-red-400 font-semibold">API OFFLINE</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
