import React from 'react';
import { HardDrive, Clock, Radio, Layers, Cpu, FileCheck } from 'lucide-react';
import { AudioMetadata } from '@/lib/types';

interface AudioMetadataCardProps {
  metadata: AudioMetadata;
}

export const AudioMetadataCard: React.FC<AudioMetadataCardProps> = ({ metadata }) => {
  const origSr = metadata.metadata?.original_sample_rate || metadata.sample_rate;
  const origCh = metadata.metadata?.original_channels || metadata.channels;
  const format = (metadata.metadata?.format || 'wav').toUpperCase();

  return (
    <div className="forensic-panel p-6 space-y-4">
      <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
        <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-slate-200 flex items-center space-x-2">
          <HardDrive className="w-4 h-4 text-cyan-400" />
          <span>Decoded Audio Metadata</span>
        </h3>
        <span className="px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/30 text-cyan-300 font-mono text-[10px] uppercase font-bold">
          {format}
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 font-mono text-xs">
        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">FORMAT</span>
          <span className="text-white font-bold text-xs">{format}</span>
        </div>

        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">DURATION</span>
          <span className="text-white font-bold text-xs">{metadata.duration_seconds.toFixed(2)}s</span>
        </div>

        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">SAMPLE RATE</span>
          <span className="text-white font-bold text-xs">{metadata.sample_rate.toLocaleString('en-US')} Hz</span>
          {origSr !== metadata.sample_rate && (
            <span className="block text-[10px] text-slate-500">from {origSr.toLocaleString('en-US')} Hz</span>
          )}
        </div>

        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">CHANNELS</span>
          <span className="text-white font-bold text-xs">Mono (1 ch)</span>
          {origCh > 1 && (
            <span className="block text-[10px] text-slate-500">from {origCh} ch</span>
          )}
        </div>

        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">FRAMES</span>
          <span className="text-white font-bold text-xs">{metadata.num_frames.toLocaleString('en-US')}</span>
        </div>

        <div className="bg-[#0b0f17] p-3.5 rounded-xl border border-white/[0.06]">
          <span className="text-slate-400 text-[10px] block uppercase mb-1">DTYPE</span>
          <span className="text-cyan-300 font-bold text-xs">{metadata.metadata?.dtype || 'float32'}</span>
        </div>
      </div>
    </div>
  );
};
