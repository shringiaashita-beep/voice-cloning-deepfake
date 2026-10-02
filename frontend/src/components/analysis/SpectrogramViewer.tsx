'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Layers, Crosshair } from 'lucide-react';
import { SpectrogramPayload } from '@/lib/types';

interface SpectrogramViewerProps {
  spectrogram: SpectrogramPayload;
  durationSeconds?: number;
}

interface HoverInspection {
  freqHz: number;
  timeSec: number;
  normalizedVal: number;
  dbVal: number;
  x: number;
  y: number;
}

export const SpectrogramViewer: React.FC<SpectrogramViewerProps> = ({ spectrogram, durationSeconds = 1.0 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverInspection | null>(null);

  const grid = spectrogram.data_grid || [];
  const nFreq = spectrogram.freq_bins || grid.length || 40;
  const nTime = spectrogram.time_bins || (grid[0] ? grid[0].length : 120);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (nFreq === 0 || nTime === 0) return;

    const cellWidth = canvas.width / nTime;
    const cellHeight = canvas.height / nFreq;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Render normalized matrix with forensic cyan/blue spectrum colormap
    for (let f = 0; f < nFreq; f++) {
      // Invert row index so low frequencies are at bottom (f = 0 at bottom)
      const y = canvas.height - (f + 1) * cellHeight;
      const row = grid[f] || [];

      for (let t = 0; t < nTime; t++) {
        const x = t * cellWidth;
        const val = Math.min(1.0, Math.max(0.0, row[t] || 0.0));

        // Interpolate color: Dark Navy (0.0) -> Cyan (0.7) -> Bright White-Cyan (1.0)
        let r = Math.round(7 + val * 150);
        let g = Math.round(14 + val * 230);
        let b = Math.round(27 + val * 228);

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x, y, cellWidth + 0.5, cellHeight + 0.5);
      }
    }
  }, [spectrogram, nFreq, nTime]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || nFreq === 0 || nTime === 0) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const timeRatio = Math.max(0, Math.min(1, mouseX / rect.width));
    const freqRatio = Math.max(0, Math.min(1, 1 - mouseY / rect.height));

    const tIdx = Math.min(nTime - 1, Math.floor(timeRatio * nTime));
    const fIdx = Math.min(nFreq - 1, Math.floor(freqRatio * nFreq));

    const normVal = grid[fIdx]?.[tIdx] ?? 0.0;
    const freqHz = Math.round(freqRatio * 8000);
    const timeSec = Number((timeRatio * durationSeconds).toFixed(2));
    const dbVal = Number((spectrogram.min_db + normVal * (spectrogram.max_db - spectrogram.min_db)).toFixed(1));

    setHoverInfo({
      freqHz,
      timeSec,
      normalizedVal: Number(normVal.toFixed(3)),
      dbVal,
      x: mouseX,
      y: mouseY
    });
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  return (
    <div className="forensic-card rounded-xl p-6 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold font-mono uppercase tracking-wider text-slate-200 flex items-center space-x-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Audio Spectrogram (Frequency Heat Map)</span>
          </h3>
          <p className="text-[11px] text-slate-400 font-sans mt-0.5">
            💡 Maps voice pitch over time (Bottom = Bass, Top = Treble). Bright colors show stronger voice energy.
          </p>
        </div>

        {/* Hover Inspector Telemetry Badge */}
        {hoverInfo ? (
          <div className="flex items-center space-x-3 text-xs font-mono bg-cyan-950/80 border border-cyan-500/40 px-3 py-1 rounded-md text-cyan-300">
            <Crosshair className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
            <span>Time: <strong>{hoverInfo.timeSec}s</strong></span>
            <span>Freq: <strong>{hoverInfo.freqHz} Hz</strong></span>
            <span>Power: <strong>{hoverInfo.dbVal} dB</strong></span>
          </div>
        ) : (
          <span className="text-xs text-slate-400 font-mono">
            Range: {spectrogram.min_db} dB to {spectrogram.max_db} dB (Hover for HUD)
          </span>
        )}
      </div>

      <div className="relative w-full bg-slate-950/90 rounded-lg p-2 border border-slate-800/80">
        <div className="flex">
          {/* Frequency Axis Label */}
          <div className="flex flex-col justify-between text-[10px] text-slate-400 font-mono pr-2 py-1 select-none">
            <span>8 kHz</span>
            <span>4 kHz</span>
            <span>0 Hz</span>
          </div>

          <div className="flex-1 relative">
            <canvas
              ref={canvasRef}
              width={800}
              height={180}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
              className="w-full h-44 block rounded cursor-crosshair"
              title="Hover over grid for frequency & power telemetry"
            />
          </div>
        </div>

        {/* Colorbar legend */}
        <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-800/60 font-mono text-[10px] text-slate-400 px-8">
          <span>Min Power ({spectrogram.min_db} dB)</span>
          <div className="w-48 h-2 rounded bg-gradient-to-r from-[#070e1b] via-[#0ea5e9] to-[#06b6d4] border border-slate-700"></div>
          <span>Max Power ({spectrogram.max_db} dB)</span>
        </div>
      </div>
    </div>
  );
};
