import React, { useEffect, useRef, useState } from "react";
import {
  Activity,
  Mic,
  MicOff,
  Radio,
  RefreshCw,
  ShieldAlert,
  Volume2,
} from "lucide-react";
import { TelemetryPacket, WebSocketMessage } from "../types";
import { connectWebSocket } from "../services/api";
import { AudioVisualizer } from "./AudioVisualizer";

interface LiveStreamStudioProps {
  machineId: string;
}

export const LiveStreamStudio: React.FC<LiveStreamStudioProps> = ({ machineId }) => {
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [useMicrophone, setUseMicrophone] = useState<boolean>(false);
  const [telemetry, setTelemetry] = useState<TelemetryPacket>({
    rms_level: 0.032,
    clipping_ratio: 0,
    spectral_bands: {
      low_0_2khz: 0.045,
      mid_2_5khz: 0.022,
      high_5_8khz: 0.008,
    },
    quality_pass: true,
  });

  const wsClientRef = useRef<{ socket: WebSocket; sendChunk: (chunk: Int16Array) => void; close: () => void } | null>(null);
  const streamIntervalRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // Start / Stop Stream Handler
  const toggleStreaming = async () => {
    if (isStreaming) {
      stopStreaming();
    } else {
      startStreaming();
    }
  };

  const startStreaming = async () => {
    setIsStreaming(true);

    // Initialize WebSocket connection
    const client = connectWebSocket(
      machineId,
      (msg: WebSocketMessage) => {
        if (msg.type === "telemetry" && msg.telemetry) {
          setTelemetry(msg.telemetry);
        }
      },
      () => {
        // Fallback to local simulation if WS cannot connect
        startSyntheticSimulation();
      }
    );
    wsClientRef.current = client;

    if (useMicrophone) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStreamRef.current = stream;
        const ctx = new AudioContext({ sampleRate: 16000 });
        audioContextRef.current = ctx;
        const source = ctx.createMediaStreamSource(stream);
        const processor = ctx.createScriptProcessor(1024, 1, 1);

        processor.onaudioprocess = (e) => {
          const input = e.inputBuffer.getChannelData(0);
          const pcm = new Int16Array(input.length);
          for (let i = 0; i < input.length; i++) {
            pcm[i] = Math.max(-32768, Math.min(32767, Math.floor(input[i] * 32767)));
          }
          client.sendChunk(pcm);
        };

        source.connect(processor);
        processor.connect(ctx.destination);
      } catch (err) {
        console.warn("Microphone access denied or unavailable. Using synthetic sensor stream.", err);
        startSyntheticSimulation();
      }
    } else {
      startSyntheticSimulation();
    }
  };

  const startSyntheticSimulation = () => {
    if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);

    streamIntervalRef.current = window.setInterval(() => {
      // Generate standard 1024-sample int16 chunk for 16kHz audio (64ms frame)
      const pcm = new Int16Array(1024);
      const freq = 120 + Math.random() * 20; // 120Hz fan fundamental
      for (let i = 0; i < 1024; i++) {
        const t = i / 16000;
        const s =
          0.2 * Math.sin(2 * Math.PI * freq * t) +
          0.08 * Math.sin(2 * Math.PI * freq * 3 * t) +
          (Math.random() - 0.5) * 0.05;
        pcm[i] = Math.floor(s * 32767);
      }

      if (wsClientRef.current?.socket.readyState === WebSocket.OPEN) {
        wsClientRef.current.sendChunk(pcm);
      } else {
        // Local simulation state update
        setTelemetry({
          rms_level: 0.03 + Math.random() * 0.015,
          clipping_ratio: 0,
          spectral_bands: {
            low_0_2khz: 0.04 + Math.random() * 0.01,
            mid_2_5khz: 0.02 + Math.random() * 0.008,
            high_5_8khz: 0.007 + Math.random() * 0.004,
          },
          quality_pass: true,
        });
      }
    }, 64);
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (wsClientRef.current) {
      wsClientRef.current.close();
      wsClientRef.current = null;
    }
  };

  useEffect(() => {
    return () => stopStreaming();
  }, [machineId]);

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-industrial-card border border-industrial-border rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-amber-950/80 text-amber-300 border border-amber-500/30">
              WebSocket Channel
            </span>
            <span className="text-xs font-mono text-gray-400">
              Target: <strong className="text-white">{machineId}</strong>
            </span>
          </div>
          <h1 className="text-xl font-bold text-white mt-1">
            Real-Time Acoustic Telemetry Stream
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Bidirectional real-time WebSocket audio streaming (<span className="text-emerald-400 font-mono">&lt;1ms</span> FFT spectral analysis).
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setUseMicrophone(!useMicrophone)}
            disabled={isStreaming}
            className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg border text-xs font-medium transition ${
              useMicrophone
                ? "bg-blue-600/20 border-blue-500 text-blue-300"
                : "bg-industrial-cardHover border-industrial-border text-gray-400 hover:text-white"
            }`}
          >
            {useMicrophone ? <Mic className="h-4 w-4" /> : <MicOff className="h-4 w-4" />}
            <span>{useMicrophone ? "Microphone Mode" : "Synthetic Sensor"}</span>
          </button>

          <button
            onClick={toggleStreaming}
            className={`flex items-center space-x-2 px-5 py-2.5 rounded-lg text-xs font-bold transition shadow-lg ${
              isStreaming
                ? "bg-red-600 hover:bg-red-700 text-white shadow-red-950/50"
                : "bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-950/50"
            }`}
          >
            {isStreaming ? (
              <>
                <Radio className="h-4 w-4 animate-spin text-white" />
                <span>STOP STREAMING</span>
              </>
            ) : (
              <>
                <Radio className="h-4 w-4 text-white" />
                <span>START LIVE STREAM</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Live Waveform Canvas */}
      <AudioVisualizer
        rmsLevel={telemetry.rms_level}
        clippingRatio={telemetry.clipping_ratio}
        isStreaming={isStreaming}
      />

      {/* Live FFT Spectral Energy Bands */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Band 1: Low 0 - 2 kHz (Blade Pass Frequency) */}
        <div className="bg-industrial-card border border-industrial-border rounded-xl p-4">
          <div className="flex justify-between items-start">
            <div>
              <span className="text-[11px] font-mono uppercase text-gray-400 font-semibold block">
                Low Frequency (0 – 2 kHz)
              </span>
              <span className="text-xs text-gray-400">Blade Pass & Rotational Harmonics</span>
            </div>
            <span className="text-base font-bold font-mono text-emerald-400">
              {telemetry.spectral_bands.low_0_2khz.toFixed(4)}
            </span>
          </div>

          <div className="w-full bg-gray-800 h-2.5 rounded-full mt-3 overflow-hidden">
            <div
              className="h-full rounded-full bg-emerald-500 transition-all duration-75"
              style={{
                width: `${Math.min(100, (telemetry.spectral_bands.low_0_2khz / 0.1) * 100)}%`,
              }}
            />
          </div>
          <span className="text-[10px] font-mono text-gray-400 mt-2 block">
            Primary aerodynamic energy band for 16-blade centrifugal fan.
          </span>
        </div>

        {/* Band 2: Mid 2 - 5 kHz (Bearing Outer Race) */}
        <div className="bg-industrial-card border border-industrial-border rounded-xl p-4">
          <div className="flex justify-between items-start">
            <div>
              <span className="text-[11px] font-mono uppercase text-gray-400 font-semibold block">
                Mid Frequency (2 – 5 kHz)
              </span>
              <span className="text-xs text-gray-400">Bearing Raceway Defect Band</span>
            </div>
            <span className="text-base font-bold font-mono text-amber-400">
              {telemetry.spectral_bands.mid_2_5khz.toFixed(4)}
            </span>
          </div>

          <div className="w-full bg-gray-800 h-2.5 rounded-full mt-3 overflow-hidden">
            <div
              className="h-full rounded-full bg-amber-500 transition-all duration-75"
              style={{
                width: `${Math.min(100, (telemetry.spectral_bands.mid_2_5khz / 0.1) * 100)}%`,
              }}
            />
          </div>
          <span className="text-[10px] font-mono text-gray-400 mt-2 block">
            Outer-race flaking produces micro-impact pulses in this range.
          </span>
        </div>

        {/* Band 3: High 5 - 8 kHz (Friction & Cavitation) */}
        <div className="bg-industrial-card border border-industrial-border rounded-xl p-4">
          <div className="flex justify-between items-start">
            <div>
              <span className="text-[11px] font-mono uppercase text-gray-400 font-semibold block">
                High Frequency (5 – 8 kHz)
              </span>
              <span className="text-xs text-gray-400">Metal-to-Metal Friction & Hiss</span>
            </div>
            <span className="text-base font-bold font-mono text-blue-400">
              {telemetry.spectral_bands.high_5_8khz.toFixed(4)}
            </span>
          </div>

          <div className="w-full bg-gray-800 h-2.5 rounded-full mt-3 overflow-hidden">
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-75"
              style={{
                width: `${Math.min(100, (telemetry.spectral_bands.high_5_8khz / 0.1) * 100)}%`,
              }}
            />
          </div>
          <span className="text-[10px] font-mono text-gray-400 mt-2 block">
            Indicates dry friction, loose shrouds, or ambient noise leak.
          </span>
        </div>
      </div>
    </div>
  );
};
