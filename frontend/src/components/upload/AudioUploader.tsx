'use client';

import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, FileAudio, X, AlertTriangle, ArrowRight, FolderPlus, Mic, Square, Sparkles, Activity } from 'lucide-react';
import { DegradationMode, DEGRADATION_PROFILES, applyDegradationFilter } from '@/lib/audioDegradation';
import { AudioDegradationFilter } from './AudioDegradationFilter';

interface AudioUploaderProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
}

const ALLOWED_EXTENSIONS = [
  '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac',
  '.webm', '.opus', '.wma', '.aiff', '.aif', '.aifc',
  '.caf', '.amr', '.3gp', '.3gpp', '.mp4', '.oga'
];
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

export const AudioUploader: React.FC<AudioUploaderProps> = ({ onFileSelect, isLoading }) => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [degradationFilter, setDegradationFilter] = useState<DegradationMode>('none');
  const [isProcessingFilter, setIsProcessingFilter] = useState<boolean>(false);

  // 5-Second In-Browser Live Microphone Stage Recorder state
  const [isStageRecording, setIsStageRecording] = useState<boolean>(false);
  const [stageCountdown, setStageCountdown] = useState<number>(5);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Clean up recording resources on unmount
  useEffect(() => {
    return () => {
      cleanupStageRecorder();
    };
  }, []);

  const cleanupStageRecorder = () => {
    if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
    }
  };

  const drawStageVisualizer = () => {
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

      // Glowing dynamic waveform
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#ef4444'; // Red recording pulse
      ctx.shadowBlur = 10;
      ctx.shadowColor = '#ef4444';

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

      // Frequency accent bars
      const freqArray = new Uint8Array(32);
      analyser.getByteFrequencyData(freqArray);
      const barWidth = canvas.width / 32;

      for (let i = 0; i < 32; i++) {
        const barHeight = (freqArray[i] / 255) * (canvas.height * 0.45);
        ctx.fillStyle = 'rgba(6, 182, 212, 0.35)'; // Cyan accent
        ctx.fillRect(i * barWidth, canvas.height - barHeight, barWidth - 1, barHeight);
      }
    };

    render();
  };

  const start5sStageRecording = async () => {
    setValidationError(null);
    setSelectedFile(null);
    audioChunksRef.current = [];
    setStageCountdown(5);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: false,
        },
      });

      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyserRef.current = analyser;

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

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || 'audio/wav' });
        if (blob.size < 800) {
          setValidationError('Recording was too short. Please speak continuously into your microphone.');
          setIsStageRecording(false);
          return;
        }

        const ext = recorder.mimeType.includes('ogg') ? 'ogg' : (recorder.mimeType.includes('webm') ? 'webm' : 'wav');
        let file = new File([blob], `live_stage_sample_${Date.now()}.${ext}`, {
          type: recorder.mimeType || 'audio/wav',
        });

        // If degradation filter is active, apply before analyzing
        if (degradationFilter !== 'none') {
          try {
            setIsProcessingFilter(true);
            const degraded = await applyDegradationFilter(file, degradationFilter);
            file = degraded.file;
          } catch {
            // fallback to original if filter fails
          } finally {
            setIsProcessingFilter(false);
          }
        }

        setIsStageRecording(false);
        // Instantly trigger analysis!
        onFileSelect(file);
      };

      recorder.start(100);
      setIsStageRecording(true);
      drawStageVisualizer();

      let remaining = 5;
      countdownIntervalRef.current = setInterval(() => {
        remaining -= 1;
        setStageCountdown(remaining);
        if (remaining <= 0) {
          if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
          }
        }
      }, 1000);
    } catch (err) {
      setValidationError(
        err instanceof Error ? err.message : 'Microphone access denied or not supported by your browser.'
      );
      setIsStageRecording(false);
    }
  };

  const stopStageRecordingEarly = () => {
    if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  };

  const cancelStageRecording = () => {
    cleanupStageRecorder();
    setIsStageRecording(false);
    audioChunksRef.current = [];
  };

  const validateFile = (file: File): boolean => {
    setValidationError(null);

    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setValidationError(`Unsupported format "${ext}". Supported formats: WAV, MP3, M4A, AAC, FLAC, OGG, WEBM, and all standard audio formats.`);
      return false;
    }

    if (file.size <= 0) {
      setValidationError('The selected file is empty (0 bytes). Please choose a valid audio recording.');
      return false;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      const mbSize = (file.size / (1024 * 1024)).toFixed(2);
      setValidationError(`File size (${mbSize} MB) exceeds maximum allowed limit of 10 MB.`);
      return false;
    }

    return true;
  };

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (validateFile(file)) {
      setSelectedFile(file);
    } else {
      setSelectedFile(null);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    handleFiles(e.target.files);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async () => {
    if (selectedFile && validateFile(selectedFile)) {
      let fileToAnalyze = selectedFile;
      if (degradationFilter !== 'none') {
        try {
          setIsProcessingFilter(true);
          const degraded = await applyDegradationFilter(selectedFile, degradationFilter);
          fileToAnalyze = degraded.file;
        } catch {
          // fallback to original
        } finally {
          setIsProcessingFilter(false);
        }
      }
      onFileSelect(fileToAnalyze);
    }
  };

  return (
    <div className="w-full space-y-4">
      {/* Hidden Native File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.webm,.opus,.wma,.aiff,.aif,.caf,.amr,.3gp,.mp4,audio/*,video/webm,video/mp4"
        onChange={handleChange}
        className="hidden"
        disabled={isLoading || isStageRecording}
      />

      {/* Mode 1: Active 5-Second In-Browser Live Microphone Stage Recorder */}
      {isStageRecording ? (
        <div className="relative rounded-[20px] p-6 border-2 border-red-500/80 bg-[#090d18] shadow-2xl shadow-red-500/20 flex flex-col items-center justify-center space-y-4 animate-fadeIn">
          {/* Header */}
          <div className="flex items-center justify-between w-full border-b border-red-500/30 pb-3">
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-red-500 animate-ping" />
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 -ml-4" />
              <span className="text-xs font-mono font-bold text-red-400 tracking-wider uppercase">
                Live Stage Mic Recording (Pitch Demo Mode)
              </span>
            </div>
            <span className="text-xs font-mono font-bold text-white px-2.5 py-0.5 rounded-full bg-red-950/80 border border-red-500/40">
              Auto-Analyze in {stageCountdown}s
            </span>
          </div>

          {/* Visualizer Canvas */}
          <div className="w-full h-32 rounded-xl overflow-hidden border border-red-500/30 bg-[#050811] relative">
            <canvas ref={canvasRef} width={640} height={128} className="w-full h-full block" />
            <div className="absolute top-2 right-3 font-mono text-[11px] text-red-400 font-bold bg-black/60 px-2 py-0.5 rounded border border-red-500/30">
              16,000 Hz · Mono
            </div>
          </div>

          {/* Instruction & Countdown Banner */}
          <div className="text-center space-y-1">
            <div className="text-2xl font-black font-mono text-white tracking-tight flex items-center justify-center space-x-2">
              <Mic className="w-5 h-5 text-red-400 animate-bounce" />
              <span>Speak for 5 Seconds on Stage...</span>
            </div>
            <p className="text-xs text-slate-300">
              Say: <span className="italic text-cyan-300">&quot;Hello, this is my authentic voice for the VoxGuard forensic jury.&quot;</span>
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex items-center space-x-3 pt-2">
            <button
              type="button"
              onClick={stopStageRecordingEarly}
              className="px-5 py-2.5 rounded-xl bg-red-600 hover:bg-red-500 text-white font-mono font-bold text-xs flex items-center space-x-2 transition-all shadow-lg shadow-red-600/30 cursor-pointer"
            >
              <Square className="w-4 h-4 fill-current" />
              <span>Finish & Analyze Now ({stageCountdown}s)</span>
            </button>
            <button
              type="button"
              onClick={cancelStageRecording}
              className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-xs transition-colors cursor-pointer"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : !selectedFile ? (
        /* Mode 2: Idle File Upload & Record Trigger */
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          className={`relative min-h-[300px] rounded-[20px] p-6 sm:p-8 text-center flex flex-col items-center justify-center transition-all duration-200 ${
            dragActive
              ? 'border-2 border-cyan-400 bg-cyan-950/20 shadow-xl shadow-cyan-500/10'
              : 'border border-dashed border-white/20 hover:border-cyan-500/50 bg-[#101622]/80 hover:bg-[#131b2b]/90'
          }`}
        >
          <div className="flex flex-col items-center justify-center space-y-4 max-w-md">
            {/* Circular Upload Icon */}
            <div className="w-16 h-16 rounded-full bg-slate-900/90 border border-white/10 flex items-center justify-center text-cyan-400 shadow-md">
              <UploadCloud className="w-8 h-8" />
            </div>

            <div className="space-y-1">
              <h3 className="text-lg font-bold text-white tracking-tight">
                Drop your audio file here
              </h3>
              <p className="text-xs text-slate-400">
                Browse audio from your system or record live from your microphone for the jury
              </p>
            </div>

            {/* Side-by-Side Action Buttons: Browse File & Record from Mic (5s Live Demo) */}
            <div className="flex flex-wrap items-center justify-center gap-3 pt-1">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                className="px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs font-sans flex items-center space-x-2 transition-all shadow-lg shadow-cyan-600/20 cursor-pointer disabled:opacity-50"
              >
                <FolderPlus className="w-4 h-4" />
                <span>Browse Audio</span>
              </button>

              <button
                type="button"
                onClick={start5sStageRecording}
                disabled={isLoading}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 text-white font-bold text-xs font-sans flex items-center space-x-2 transition-all shadow-lg shadow-red-600/25 cursor-pointer disabled:opacity-50 group"
              >
                <Mic className="w-4 h-4 group-hover:scale-110 transition-transform" />
                <span>Record from Mic (5s Pitch)</span>
              </button>
            </div>

            {/* Metadata Row */}
            <div className="text-[11px] font-mono text-slate-400 pt-3 border-t border-white/[0.06] w-full flex flex-col sm:flex-row items-center justify-between gap-1">
              <span>Formats: WAV · MP3 · MP4 · M4A · AAC · FLAC · OGG · WEBM</span>
              <span>Limit: 10 MB · Max 120s</span>
            </div>
          </div>
        </div>
      ) : (
        /* Mode 3: Selected File Ready State */
        <div className="forensic-panel p-6 border border-white/10 rounded-[20px] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3.5 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400">
                <FileAudio className="w-7 h-7" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-white font-mono truncate max-w-md">
                  {selectedFile.name}
                </h4>
                <div className="flex items-center space-x-3 text-xs text-slate-400 font-mono mt-1">
                  <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                  <span>•</span>
                  <span className="uppercase font-semibold text-cyan-400 px-2 py-0.5 rounded bg-white/[0.06] border border-white/10">
                    {selectedFile.name.split('.').pop()}
                  </span>
                  {degradationFilter !== 'none' && (
                    <>
                      <span>•</span>
                      <span className="px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/40 text-amber-300 font-bold">
                        Filter: {degradationFilter === 'whatsapp' ? 'WhatsApp Opus' : '8kHz Phone'}
                      </span>
                    </>
                  )}
                </div>
              </div>
            </div>

            <button
              onClick={handleClear}
              disabled={isLoading || isProcessingFilter}
              className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
              title="Remove file"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="pt-2 flex justify-end space-x-3">
            <button
              onClick={handleClear}
              disabled={isLoading || isProcessingFilter}
              className="px-4 py-2 rounded-xl bg-white/[0.06] hover:bg-white/10 text-slate-300 font-sans text-xs transition-colors cursor-pointer"
            >
              Remove
            </button>
            <button
              onClick={handleSubmit}
              disabled={isLoading || isProcessingFilter}
              className="px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs flex items-center space-x-2 transition-all shadow-lg shadow-cyan-600/20 disabled:opacity-50 cursor-pointer"
            >
              <span>{isProcessingFilter ? 'Applying Filter...' : 'Analyze Audio'}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Degradation Filter Selector (Requirement 4) */}
      <div className="pt-1">
        <AudioDegradationFilter
          selectedFilter={degradationFilter}
          onSelectFilter={setDegradationFilter}
          disabled={isLoading || isStageRecording}
        />
      </div>

      {/* Validation Error Alert */}
      {validationError && (
        <div className="p-3.5 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-300 text-xs font-mono flex items-center space-x-2.5">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>{validationError}</span>
        </div>
      )}
    </div>
  );
};
