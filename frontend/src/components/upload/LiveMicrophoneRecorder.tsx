'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, Pause, RotateCcw, Shield, AlertCircle, Sparkles } from 'lucide-react';

interface LiveMicrophoneRecorderProps {
  onRecordingComplete: (file: File) => void;
  isLoading: boolean;
}

export const LiveMicrophoneRecorder: React.FC<LiveMicrophoneRecorderProps> = ({
  onRecordingComplete,
  isLoading,
}) => {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingTime, setRecordingTime] = useState<number>(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [recordedFile, setRecordedFile] = useState<File | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopRecording();
      if (timerRef.current) clearInterval(timerRef.current);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, []);

  // Web Audio visualizer loop
  const drawVisualizer = () => {
    if (!analyserRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      animFrameRef.current = requestAnimationFrame(render);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = '#050811';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Glowing center wave
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#06b6d4';
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#06b6d4';

      ctx.beginPath();
      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();

      // Secondary frequency accent bars
      const freqArray = new Uint8Array(32);
      analyser.getByteFrequencyData(freqArray);
      const barWidth = canvas.width / 32;

      for (let i = 0; i < 32; i++) {
        const barHeight = (freqArray[i] / 255) * (canvas.height * 0.4);
        ctx.fillStyle = 'rgba(16, 185, 129, 0.25)';
        ctx.fillRect(i * barWidth, canvas.height - barHeight, barWidth - 1, barHeight);
      }
    };

    render();
  };

  const startRecording = async (maxDuration: number = 30) => {
    setPermissionError(null);
    setAudioUrl(null);
    setRecordedFile(null);
    setRecordingTime(0);
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      // Audio Context for real-time visualization
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyserRef.current = analyser;

      drawVisualizer();

      // MediaRecorder
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus') ? 'audio/ogg;codecs=opus' : '');

      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || 'audio/wav' });

        // Clean up tracks & animations
        stream.getTracks().forEach((track) => track.stop());
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

        if (blob.size === 0) {
          setPermissionError('No audio signal captured from microphone. Please ensure your microphone is active and unmuted.');
          setRecordedFile(null);
          setAudioUrl(null);
          return;
        }

        const url = URL.createObjectURL(blob);
        setAudioUrl(url);

        const ext = recorder.mimeType.includes('ogg') ? 'ogg' : (recorder.mimeType.includes('webm') ? 'webm' : 'wav');
        const file = new File([blob], `live_mic_capture_${Date.now()}.${ext}`, {
          type: recorder.mimeType || 'audio/wav',
        });
        setRecordedFile(file);

        // Instantly trigger voice deepfake forensic analysis!
        onRecordingComplete(file);
      };

      recorder.start(100);
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => {
          if (maxDuration > 0 && prev >= maxDuration - 1) {
            stopRecording();
            return maxDuration;
          }
          return prev + 1;
        });
      }, 1000);
    } catch (err) {
      setPermissionError(
        err instanceof Error ? err.message : 'Microphone access was denied or not supported by your browser.'
      );
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) clearInterval(timerRef.current);
    setIsRecording(false);
  };

  const handleReset = () => {
    stopRecording();
    setAudioUrl(null);
    setRecordedFile(null);
    setRecordingTime(0);
    setIsPlaying(false);
  };

  const togglePlayback = () => {
    if (!audioPlayerRef.current) return;
    if (isPlaying) {
      audioPlayerRef.current.pause();
      setIsPlaying(false);
    } else {
      audioPlayerRef.current.play();
      setIsPlaying(true);
    }
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="bg-[#0b101c] border border-cyan-500/25 rounded-2xl p-6 space-y-5 shadow-xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <Mic className="w-4 h-4 text-cyan-400" />
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
              Live Microphone Voice Capture
            </h4>
          </div>
          <p className="text-xs text-slate-400 mt-0.5 font-sans">
            Record voice live from your microphone with real-time acoustic wave monitoring.
          </p>
        </div>
        <div className="flex items-center space-x-2">
          {isRecording && (
            <span className="flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-red-950/80 border border-red-500/40 text-red-400 text-[10px] font-mono font-bold animate-pulse">
              <span className="w-2 h-2 rounded-full bg-red-500" />
              <span>RECORDING {formatTime(recordingTime)} (UNLIMITED DURATION)</span>
            </span>
          )}
          <span className="px-2 py-0.5 rounded-full bg-emerald-950/60 border border-emerald-500/30 text-emerald-400 text-[10px] font-mono font-bold">
            HUMAN VOICE DETECTOR ACTIVE
          </span>
        </div>
      </div>

      {/* Permission Error Message */}
      {permissionError && (
        <div className="p-3.5 rounded-xl bg-red-950/40 border border-red-500/30 text-red-300 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{permissionError}</span>
        </div>
      )}

      {/* Real-Time Wave Visualizer Canvas */}
      <div className="relative rounded-xl overflow-hidden border border-slate-800/80 bg-[#050811] h-32 flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={640}
          height={128}
          className="w-full h-full block"
        />

        {!isRecording && !audioUrl && (
          <div className="absolute inset-0 flex flex-col items-center justify-center space-y-1 select-none pointer-events-none bg-slate-950/40">
            <Mic className="w-6 h-6 text-slate-600" />
            <span className="text-xs font-mono text-slate-500">
              Ready to capture any length. Speak into your microphone.
            </span>
          </div>
        )}
      </div>

      {/* Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 pt-1">
        <div className="flex items-center space-x-3">
          {!isRecording ? (
            <>
              <button
                onClick={() => startRecording(0)}
                disabled={isLoading}
                className="px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs font-mono flex items-center space-x-2 transition-all shadow-lg shadow-cyan-600/20 cursor-pointer disabled:opacity-50"
              >
                <Mic className="w-4 h-4" />
                <span>Unlimited Record Mode</span>
              </button>
              <button
                onClick={() => startRecording(0)}
                disabled={isLoading}
                className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-slate-950 font-bold text-xs font-mono flex items-center space-x-2 transition-all shadow-lg shadow-emerald-600/25 cursor-pointer disabled:opacity-50"
              >
                <Sparkles className="w-4 h-4" />
                <span>Capture Any Duration Voice</span>
              </button>
            </>
          ) : (
            <button
              onClick={stopRecording}
              className="px-5 py-2.5 rounded-xl bg-red-600 hover:bg-red-500 text-white font-bold text-xs font-mono flex items-center space-x-2 transition-all shadow-lg shadow-red-600/20 cursor-pointer animate-pulse"
            >
              <Square className="w-4 h-4 fill-current" />
              <span>Stop Recording ({formatTime(recordingTime)})</span>
            </button>
          )}

          {audioUrl && (
            <>
              <button
                onClick={togglePlayback}
                className="px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-cyan-300 font-mono text-xs flex items-center space-x-1.5 transition-colors cursor-pointer"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                <span>{isPlaying ? 'Pause' : 'Preview'}</span>
              </button>

              <button
                onClick={handleReset}
                className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-white transition-colors cursor-pointer"
                title="Reset Recording"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>

              <audio
                ref={audioPlayerRef}
                src={audioUrl}
                onEnded={() => setIsPlaying(false)}
                className="hidden"
              />
            </>
          )}
        </div>

        {/* Action Button: Analyze Captured Audio */}
        {recordedFile && (
          <button
            onClick={() => onRecordingComplete(recordedFile)}
            disabled={isLoading}
            className="px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold text-xs font-mono flex items-center space-x-2 transition-all shadow-lg shadow-emerald-600/20 cursor-pointer disabled:opacity-50 ml-auto"
          >
            <Shield className="w-4 h-4" />
            <span>Analyze Live Recording</span>
          </button>
        )}
      </div>
    </div>
  );
};
