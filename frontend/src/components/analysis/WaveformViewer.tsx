'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Activity, Play, Pause, RotateCcw, Volume2 } from 'lucide-react';
import { WaveformPayload } from '@/lib/types';

interface WaveformViewerProps {
  waveform: WaveformPayload;
  audioFile?: File | null;
}

export const WaveformViewer: React.FC<WaveformViewerProps> = ({ waveform, audioFile }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Generate audio object URL when audioFile prop is provided
  useEffect(() => {
    if (audioFile) {
      const url = URL.createObjectURL(audioFile);
      setAudioUrl(url);
      return () => {
        URL.revokeObjectURL(url);
      };
    } else {
      setAudioUrl(null);
    }
  }, [audioFile]);

  // Audio Playback event listeners
  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !waveform.duration_seconds) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = clickX / rect.width;
    const seekTime = ratio * waveform.duration_seconds;

    setCurrentTime(seekTime);
    if (audioRef.current) {
      audioRef.current.currentTime = seekTime;
    }
  };

  // Render Canvas Waveform Envelope with Playback Scrubber Line
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerY = height / 2;

    ctx.clearRect(0, 0, width, height);

    // Draw horizontal zero line
    ctx.strokeStyle = 'rgba(30, 41, 59, 0.8)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    const minPeaks = waveform.min_peaks || [];
    const maxPeaks = waveform.max_peaks || [];
    const points = waveform.peak_envelope || [];
    const count = points.length;

    if (count > 0) {
      const step = width / count;

      // Fill waveform envelope area
      ctx.fillStyle = 'rgba(6, 182, 212, 0.18)';
      ctx.beginPath();

      for (let i = 0; i < count; i++) {
        const x = i * step;
        const y = centerY - maxPeaks[i] * (height / 2.2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      for (let i = count - 1; i >= 0; i--) {
        const x = i * step;
        const y = centerY - minPeaks[i] * (height / 2.2);
        ctx.lineTo(x, y);
      }

      ctx.closePath();
      ctx.fill();

      // Stroke peak amplitude vertical lines
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 1.5;
      ctx.beginPath();

      for (let i = 0; i < count; i++) {
        const x = i * step;
        const minY = centerY - minPeaks[i] * (height / 2.2);
        const maxY = centerY - maxPeaks[i] * (height / 2.2);

        ctx.moveTo(x, minY);
        ctx.lineTo(x, maxY);
      }

      ctx.stroke();
    }

    // Draw Playback Scrubber Line if audio is playing/seeking
    if (waveform.duration_seconds > 0 && currentTime > 0) {
      const scrubberX = (currentTime / waveform.duration_seconds) * width;
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(scrubberX, 0);
      ctx.lineTo(scrubberX, height);
      ctx.stroke();

      // Scrubber handle dot
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(scrubberX, 8, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [waveform, currentTime]);

  return (
    <div className="forensic-card rounded-xl p-6 space-y-4">
      {/* Hidden Audio Element */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={handleEnded}
        />
      )}

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold font-mono uppercase tracking-wider text-slate-200 flex items-center space-x-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span>Audio Waveform (Loudness Over Time)</span>
          </h3>
          <p className="text-[11px] text-slate-400 font-sans mt-0.5">
            💡 High green peaks show loud speech sounds. Quiet gaps show natural pauses or breaths.
          </p>
        </div>

        {/* Audio Playback Controls */}
        <div className="flex items-center space-x-3 font-mono text-xs">
          {audioUrl ? (
            <button
              onClick={togglePlay}
              className="px-3 py-1.5 rounded-lg bg-cyan-950/80 hover:bg-cyan-900/90 border border-cyan-500/40 text-cyan-300 flex items-center space-x-2 transition-colors font-semibold"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-3.5 h-3.5" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span>Play Recording</span>
                </>
              )}
            </button>
          ) : (
            <span className="text-slate-400 flex items-center space-x-1">
              <Volume2 className="w-3.5 h-3.5 text-slate-500" />
              <span>Interactive Scrubber</span>
            </span>
          )}

          <span className="text-slate-400">
            {currentTime.toFixed(2)}s / {waveform.duration_seconds.toFixed(2)}s
          </span>
        </div>
      </div>

      <div className="relative w-full bg-slate-950/90 rounded-lg p-2 border border-slate-800/80 cursor-pointer">
        <canvas
          ref={canvasRef}
          width={800}
          height={160}
          onClick={handleCanvasClick}
          className="w-full h-40 block rounded"
          title="Click to seek audio playback position"
        />

        {/* Time markers */}
        <div className="flex justify-between text-[10px] text-slate-400 font-mono px-2 mt-1">
          <span>0.00s</span>
          <span>{(waveform.duration_seconds / 2).toFixed(2)}s</span>
          <span>{waveform.duration_seconds.toFixed(2)}s</span>
        </div>
      </div>
    </div>
  );
};
