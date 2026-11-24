import React, { useEffect, useRef, useState } from "react";
import { Play, Pause, Volume2 } from "lucide-react";

interface AudioVisualizerProps {
  audioUrl?: string;
  rmsLevel?: number;
  clippingRatio?: number;
  isStreaming?: boolean;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  audioUrl,
  rmsLevel = 0.035,
  clippingRatio = 0,
  isStreaming = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(10);

  // Oscilloscope animation effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let phase = 0;

    const render = () => {
      ctx.fillStyle = "#0B0F17";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw subtle grid lines
      ctx.strokeStyle = "#1F2937";
      ctx.lineWidth = 1;
      for (let y = 20; y < canvas.height; y += 25) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // Draw center line
      const centerY = canvas.height / 2;
      ctx.strokeStyle = "#374151";
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(canvas.width, centerY);
      ctx.stroke();

      // Draw acoustic waveform
      ctx.lineWidth = 2;
      ctx.strokeStyle = isStreaming
        ? "#F59E0B"
        : clippingRatio > 0.05
        ? "#EF4444"
        : "#10B981";

      ctx.beginPath();
      const amplitude = Math.max(8, Math.min(centerY - 5, rmsLevel * 300));
      const freq = isStreaming ? 0.04 : 0.02;

      for (let x = 0; x < canvas.width; x++) {
        // Multi-frequency harmonic representation of fan blade pass frequency
        const harmonic1 = Math.sin(x * freq + phase);
        const harmonic2 = 0.3 * Math.sin(x * freq * 3 - phase * 1.5);
        const noise = (Math.random() - 0.5) * (rmsLevel * 10);
        const y = centerY + (harmonic1 + harmonic2) * amplitude + noise;

        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      phase += isPlaying || isStreaming ? 0.08 : 0.01;
      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [isPlaying, isStreaming, rmsLevel, clippingRatio]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  return (
    <div className="bg-industrial-card border border-industrial-border rounded-xl p-4 flex flex-col space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Volume2 className="h-4 w-4 text-emerald-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-gray-300 font-mono">
            Acoustic Oscilloscope & Audio Waveform
          </span>
        </div>
        <div className="flex items-center space-x-3 text-xs font-mono">
          <span className="text-gray-400">
            RMS: <strong className="text-white">{rmsLevel.toFixed(4)}</strong>
          </span>
          <span className={`${clippingRatio > 0.01 ? "text-red-400 font-bold" : "text-gray-400"}`}>
            Clip: {(clippingRatio * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Waveform Canvas */}
      <div className="relative rounded-lg overflow-hidden border border-gray-800 bg-[#0B0F17]">
        <canvas
          ref={canvasRef}
          width={680}
          height={140}
          className="w-full h-32 block"
        />
        {isStreaming && (
          <div className="absolute top-2 right-2 flex items-center space-x-1.5 px-2 py-0.5 rounded bg-red-950/80 border border-red-500/50 text-red-300 text-[10px] font-mono animate-pulse font-semibold">
            <span className="h-1.5 w-1.5 rounded-full bg-red-400"></span>
            <span>LIVE 16kHz</span>
          </div>
        )}
      </div>

      {/* Audio Playback Bar if audioUrl is provided */}
      {audioUrl && (
        <div className="flex items-center justify-between space-x-3 pt-1">
          <button
            onClick={togglePlay}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/40 text-emerald-300 text-xs font-medium transition"
          >
            {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            <span>{isPlaying ? "Pause Acoustic Playback" : "Listen to Recording"}</span>
          </button>

          <span className="text-xs font-mono text-gray-400">
            {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
          </span>

          <audio
            ref={audioRef}
            src={audioUrl}
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
            onEnded={() => setIsPlaying(false)}
          />
        </div>
      )}
    </div>
  );
};
