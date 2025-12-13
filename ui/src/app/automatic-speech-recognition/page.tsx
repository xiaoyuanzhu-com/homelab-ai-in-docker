"use client";

import { useState, useEffect, useRef, Suspense, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Volume2, Loader2, Mic, MicOff, Radio } from "lucide-react";
import { toast } from "sonner";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle } from "lucide-react";
import { TryLayout, ModelSelector } from "@/components/try";
import { AudioUpload, AudioPlayer, AudioRecorderControls } from "@/components/inputs";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";

interface SpeakerSegment {
  start: number;
  end: number;
  speaker: string;
}

interface TranscriptionResult {
  request_id: string;
  text?: string;
  model: string;
  language?: string | null;
  processing_time_ms: number;
  segments?: SpeakerSegment[];
  num_speakers?: number;
}

type ChoiceType = "model" | "lib";
interface ChoiceInfo {
  id: string;
  label: string;
  provider: string;
  status: string;
  type: ChoiceType;
  architecture?: string;
  supports_live_streaming: boolean;
}

interface ASRRequestBody {
  audio: string;
  model: string;
  output_format: "transcription" | "diarization";
  language?: string;
  return_timestamps?: boolean;
  min_speakers?: number;
  max_speakers?: number;
}

// Live transcription types
interface LiveTranscriptionSegment {
  speaker: number;
  text: string;
  start: string;
  end: string;
  translation?: string;
  detected_language?: string;
}

interface LiveTranscriptionMessage {
  type?: string;
  status?: string;
  lines?: LiveTranscriptionSegment[];
  buffer_transcription?: string;
  buffer_diarization?: string;
  buffer_translation?: string;
  remaining_time_transcription?: number;
  remaining_time_diarization?: number;
  error?: string;
  sampleRate?: number;
}

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";
type TranscriptionMode = "transcription" | "live";

function AutomaticSpeechRecognitionContent() {
  const searchParams = useSearchParams();

  // Engine/choice state - comes first
  const [selectedChoice, setSelectedChoice] = useState<string>("");
  const [choices, setChoices] = useState<ChoiceInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Mode state - depends on selected engine
  const [mode, setMode] = useState<TranscriptionMode>("transcription");

  // Common state
  const [language, setLanguage] = useState<string>(""); // empty = auto-detect
  const [enableDiarization, setEnableDiarization] = useState(false);
  const [minSpeakers, setMinSpeakers] = useState<string>("1");
  const [maxSpeakers, setMaxSpeakers] = useState<string>("4");
  const [isRecordingSupported, setIsRecordingSupported] = useState(false);

  // File-based transcription state
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [whisperxAsrModel, setWhisperxAsrModel] = useState<string>("large-v3");

  // File recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [rawRequest, setRawRequest] = useState<any>(null);

  // Live transcription state
  const [liveConnectionStatus, setLiveConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [isLiveStreaming, setIsLiveStreaming] = useState(false);
  const liveWsRef = useRef<WebSocket | null>(null);
  const liveStreamRef = useRef<MediaStream | null>(null);
  // AudioWorklet refs for PCM streaming
  const liveAudioContextRef = useRef<AudioContext | null>(null);
  const liveWorkletNodeRef = useRef<AudioWorkletNode | null>(null);
  const [liveModel, setLiveModel] = useState<string>("large-v3");
  const [liveLines, setLiveLines] = useState<LiveTranscriptionSegment[]>([]);
  const [liveBuffer, setLiveBuffer] = useState<string>("");
  const [liveStatus, setLiveStatus] = useState<string>("");
  const [liveError, setLiveError] = useState<string | null>(null);

  // Get the selected choice info
  const selectedChoiceInfo = choices.find(c => `${c.type}:${c.id}` === selectedChoice);
  const supportsLiveMode = selectedChoiceInfo?.supports_live_streaming ?? false;

  // Reset mode to transcription if the selected engine doesn't support live mode
  useEffect(() => {
    if (mode === "live" && !supportsLiveMode) {
      setMode("transcription");
    }
  }, [selectedChoice, supportsLiveMode, mode]);

  // Reset liveModel when switching between engine architectures
  useEffect(() => {
    const arch = selectedChoiceInfo?.architecture;
    if (arch === "funasr") {
      // Default to SenseVoice for FunASR
      setLiveModel("FunAudioLLM/SenseVoiceSmall");
    } else if (arch === "whisperlivekit" || arch === "whisperx") {
      // Default to large-v3 for Whisper-based
      setLiveModel("large-v3");
    }
  }, [selectedChoiceInfo?.architecture]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const supported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
      setIsRecordingSupported(supported);
    }

    const fetchChoices = async () => {
      try {
        const response = await fetch("/api/task-options?task=automatic-speech-recognition");
        if (!response.ok) throw new Error("Failed to fetch ASR options");
        const data = await response.json();
        interface OptionData {
          id: string;
          label: string;
          provider: string;
          status: string;
          type: string;
          architecture?: string;
          supports_live_streaming?: boolean;
        }
        const merged: ChoiceInfo[] = (data.options || [])
          .filter((o: OptionData) => o.status === "ready")
          .map((o: OptionData) => ({
            id: o.id,
            label: o.label,
            provider: o.provider,
            status: o.status,
            type: o.type as ChoiceType,
            architecture: o.architecture,
            supports_live_streaming: o.supports_live_streaming ?? false,
          }));
        setChoices(merged);

        const modelParam = searchParams.get("model");
        const libParam = searchParams.get("lib");
        const skillParam = searchParams.get("skill");
        const findChoice = (t: ChoiceType, id: string) => merged.find(c => c.type === t && c.id === id);
        if (modelParam && findChoice("model", modelParam)) {
          setSelectedChoice(`model:${modelParam}`);
        } else if (libParam && findChoice("lib", libParam)) {
          setSelectedChoice(`lib:${libParam}`);
        } else if (skillParam) {
          const m = findChoice("model", skillParam);
          const l = findChoice("lib", skillParam);
          if (m) setSelectedChoice(`model:${skillParam}`);
          else if (l) setSelectedChoice(`lib:${skillParam}`);
          else if (merged.length > 0) setSelectedChoice(`${merged[0].type}:${merged[0].id}`);
        } else if (merged.length > 0) {
          setSelectedChoice(`${merged[0].type}:${merged[0].id}`);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        toast.error("Failed to load available ASR engines");
      } finally {
        setModelsLoading(false);
      }
    };

    fetchChoices();
  }, [searchParams]);

  useEffect(() => {
    if (recordedAudioUrl) {
      setAudioPreviewUrl(recordedAudioUrl);
      return;
    }

    if (audioFile) {
      const objectUrl = URL.createObjectURL(audioFile);
      setAudioPreviewUrl(objectUrl);
      return () => {
        URL.revokeObjectURL(objectUrl);
      };
    }

    setAudioPreviewUrl(null);
    return;
  }, [audioFile, recordedAudioUrl]);

  // File recording functions
  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast.error("Your browser doesn't support audio recording. Please use a modern browser with HTTPS.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        setRecordedAudioUrl(url);
        setRecordedBlob(blob);
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        setAudioFile(file);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
      setResult(null);
      toast.success("Recording started");
    } catch (err) {
      console.error("Error accessing microphone:", err);
      toast.error("Failed to access microphone. Please grant permission and ensure you're using HTTPS.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      toast.success("Recording stopped");
    }
  };

  const clearRecording = () => {
    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl);
    }
    setRecordedAudioUrl(null);
    setRecordedBlob(null);
    setAudioFile(null);
    setError(null);
    setResult(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("audio/") && !file.name.match(/\.(mp3|mp4|mpeg|mpga|m4a|wav|webm)$/i)) {
        toast.error("Please upload an audio file (mp3, mp4, wav, webm, etc.)");
        return;
      }

      if (recordedAudioUrl) {
        URL.revokeObjectURL(recordedAudioUrl);
        setRecordedAudioUrl(null);
        setRecordedBlob(null);
      }

      setAudioFile(file);
      setError(null);
      setResult(null);
    }
  };

  const handleTranscribe = async () => {
    if (!audioFile || !selectedChoice) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64Data = (reader.result as string).split(",")[1];

        try {
          const isModel = selectedChoice.startsWith("model:");
          const isLib = selectedChoice.startsWith("lib:");
          const id = selectedChoice.split(":", 2)[1];

          let response: Response;
          let actualRequestBody: any;

          if (isModel) {
            const requestBody: ASRRequestBody = {
              audio: base64Data,
              model: id,
              output_format: enableDiarization ? "diarization" : "transcription",
            };
            if (!enableDiarization) {
              requestBody.language = language || undefined;
              requestBody.return_timestamps = false;
            }
            if (enableDiarization) {
              if (minSpeakers) requestBody.min_speakers = parseInt(minSpeakers);
              if (maxSpeakers) requestBody.max_speakers = parseInt(maxSpeakers);
            }
            actualRequestBody = requestBody;
            response = await fetch("/api/automatic-speech-recognition", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(requestBody),
            });
          } else if (isLib && id === "whisperx/whisperx") {
            const requestBody = {
              audio: base64Data,
              asr_model: whisperxAsrModel,
              language: language || undefined,
              diarize: enableDiarization,
            };
            actualRequestBody = requestBody;
            response = await fetch("/api/whisperx/transcribe", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(requestBody),
            });
          } else {
            throw new Error("Unsupported ASR engine selected");
          }

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          if (selectedChoice.startsWith("lib:") && id === "whisperx/whisperx") {
            const speakers = new Set<string>();
            if (Array.isArray(data.segments)) {
              data.segments.forEach((s: SpeakerSegment) => { if (s.speaker) speakers.add(s.speaker); });
            }
            setResult({ ...data, num_speakers: speakers.size || undefined });
          } else {
            setResult(data);
          }

          setRawRequest({
            ...actualRequestBody,
            audio: actualRequestBody.audio.substring(0, 100) + "...",
          });

          const successMsg = enableDiarization ? "Speaker analysis completed!" : "Transcription completed!";
          toast.success(successMsg, {
            description: `Processed in ${data.processing_time_ms}ms`,
          });
        } catch (err) {
          const errorMsg = err instanceof Error ? err.message : "Failed to transcribe audio";
          setError(errorMsg);
          toast.error(errorMsg);
        } finally {
          setLoading(false);
        }
      };
      reader.readAsDataURL(audioFile);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to process audio file";
      setError(errorMsg);
      toast.error(errorMsg);
      setLoading(false);
    }
  };

  // Live transcription functions
  const getLiveWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const params = new URLSearchParams();
    params.set("model", liveModel);
    params.set("language", language || "auto");
    if (enableDiarization) {
      params.set("diarization", "true");
    }

    // Use different endpoint based on selected engine architecture
    const arch = selectedChoiceInfo?.architecture;
    if (arch === "funasr") {
      return `${protocol}//${host}/api/funasr/live?${params.toString()}`;
    }
    // Default to WhisperLiveKit
    return `${protocol}//${host}/api/automatic-speech-recognition/live?${params.toString()}`;
  }, [liveModel, language, enableDiarization, selectedChoiceInfo]);

  const startLiveStreaming = useCallback(async () => {
    if (!isRecordingSupported) {
      toast.error("Your browser doesn't support audio recording");
      return;
    }

    setLiveError(null);
    setLiveLines([]);
    setLiveBuffer("");
    setLiveStatus("");
    setLiveConnectionStatus("connecting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      liveStreamRef.current = stream;

      const wsUrl = getLiveWebSocketUrl();
      console.log("Connecting to:", wsUrl);
      const ws = new WebSocket(wsUrl);
      liveWsRef.current = ws;

      ws.onopen = async () => {
        console.log("WebSocket connected");
        setLiveConnectionStatus("connected");

        try {
          // Create AudioContext and load AudioWorklet for PCM streaming
          const audioContext = new AudioContext();
          liveAudioContextRef.current = audioContext;

          // Load the PCM processor worklet
          await audioContext.audioWorklet.addModule("/pcm-processor.js");

          // Create worklet node with native sample rate info
          const workletNode = new AudioWorkletNode(audioContext, "pcm-processor", {
            processorOptions: {
              sampleRate: audioContext.sampleRate,
            },
          });
          liveWorkletNodeRef.current = workletNode;

          // Handle PCM audio chunks from worklet
          workletNode.port.onmessage = (event) => {
            if (event.data.type === "audio" && ws.readyState === WebSocket.OPEN) {
              // Send raw PCM int16 bytes
              ws.send(event.data.samples);
            }
          };

          // Connect microphone -> worklet
          const source = audioContext.createMediaStreamSource(stream);
          source.connect(workletNode);
          // Don't connect to destination (we don't want to hear ourselves)

          setIsLiveStreaming(true);
          toast.success("Live transcription started");
        } catch (workletError) {
          console.error("AudioWorklet setup failed:", workletError);
          setLiveError("Failed to initialize audio processor");
          setLiveConnectionStatus("error");
          ws.close();
        }
      };

      ws.onmessage = (event: MessageEvent) => {
        try {
          const data: LiveTranscriptionMessage = JSON.parse(event.data);

          if (data.type === "config") {
            console.log("Received config:", data);
            return;
          }

          if (data.type === "ready_to_stop") {
            console.log("Server ready to stop");
            return;
          }

          if (data.status) {
            setLiveStatus(data.status);
          }

          if (data.lines !== undefined && Array.isArray(data.lines)) {
            setLiveLines(data.lines);
          }

          if (data.buffer_transcription !== undefined) {
            setLiveBuffer(data.buffer_transcription);
          }

          if (data.error) {
            setLiveError(data.error);
          }
        } catch (err) {
          console.error("Failed to parse message:", err);
        }
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        setLiveError("WebSocket connection error");
        setLiveConnectionStatus("error");
      };

      ws.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason);
        setLiveConnectionStatus("disconnected");
        setIsLiveStreaming(false);

        // Clean up AudioWorklet
        if (liveWorkletNodeRef.current) {
          liveWorkletNodeRef.current.disconnect();
          liveWorkletNodeRef.current = null;
        }
        if (liveAudioContextRef.current) {
          liveAudioContextRef.current.close();
          liveAudioContextRef.current = null;
        }
      };
    } catch (err) {
      console.error("Failed to start streaming:", err);
      const errorMsg = err instanceof Error ? err.message : "Failed to start streaming";
      setLiveError(errorMsg);
      setLiveConnectionStatus("error");
      toast.error("Failed to start", { description: errorMsg });
    }
  }, [isRecordingSupported, getLiveWebSocketUrl]);

  const stopLiveStreaming = useCallback(() => {
    // Clean up AudioWorklet
    if (liveWorkletNodeRef.current) {
      liveWorkletNodeRef.current.port.postMessage({ type: "stop" });
      liveWorkletNodeRef.current.disconnect();
      liveWorkletNodeRef.current = null;
    }

    if (liveAudioContextRef.current) {
      liveAudioContextRef.current.close();
      liveAudioContextRef.current = null;
    }

    if (liveStreamRef.current) {
      liveStreamRef.current.getTracks().forEach((track) => track.stop());
      liveStreamRef.current = null;
    }

    if (liveWsRef.current) {
      liveWsRef.current.close();
      liveWsRef.current = null;
    }

    setIsLiveStreaming(false);
    setLiveConnectionStatus("disconnected");
    toast.info("Live transcription stopped");
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopLiveStreaming();
    };
  }, [stopLiveStreaming]);

  // Update raw request preview whenever form changes
  useEffect(() => {
    if (mode === "live") {
      setRawRequest(null);
      return;
    }

    if (!selectedChoice) {
      setRawRequest(null);
      return;
    }

    const isModel = selectedChoice.startsWith("model:");
    const isLib = selectedChoice.startsWith("lib:");
    const id = selectedChoice.split(":", 2)[1];

    if (isModel) {
      const requestBody: any = {
        audio: "<base64 audio data>",
        model: id,
        output_format: enableDiarization ? "diarization" : "transcription",
      };
      if (!enableDiarization) {
        requestBody.language = language || undefined;
        requestBody.return_timestamps = false;
      }
      if (enableDiarization) {
        if (minSpeakers) requestBody.min_speakers = parseInt(minSpeakers);
        if (maxSpeakers) requestBody.max_speakers = parseInt(maxSpeakers);
      }
      setRawRequest(requestBody);
    } else if (isLib && id === "whisperx/whisperx") {
      const requestBody = {
        audio: "<base64 audio data>",
        asr_model: whisperxAsrModel,
        language: language || undefined,
        diarize: enableDiarization,
      };
      setRawRequest(requestBody);
    }
  }, [mode, selectedChoice, enableDiarization, language, minSpeakers, maxSpeakers, whisperxAsrModel]);

  const getLiveStatusBadge = () => {
    switch (liveConnectionStatus) {
      case "connected":
        return <Badge variant="default" className="bg-green-500">Connected</Badge>;
      case "connecting":
        return <Badge variant="secondary">Connecting...</Badge>;
      case "error":
        return <Badge variant="destructive">Error</Badge>;
      default:
        return <Badge variant="outline">Disconnected</Badge>;
    }
  };

  const modelOptions = choices.map((c) => ({
    value: `${c.type}:${c.id}`,
    label: c.supports_live_streaming
      ? `${c.label} (${c.provider}) [${c.type}] ⚡ Live`
      : `${c.label} (${c.provider}) [${c.type}]`,
  }));

  const rawResponsePayload = result;
  const hasAudioInput = Boolean(audioFile || recordedBlob);

  // Unified input panel
  const inputPanel = (
    <div className="space-y-4">
      {/* Step 1: Engine Selection */}
      <div className="space-y-2">
        <Label htmlFor="engine">1. Select Engine</Label>
        <ModelSelector
          value={selectedChoice}
          onChange={setSelectedChoice}
          options={modelOptions}
          loading={modelsLoading}
          disabled={loading || isLiveStreaming}
          emptyMessage="No ASR engines ready. Visit Models/Libs to set them up."
        />
        {selectedChoiceInfo && (
          <p className="text-xs text-muted-foreground">
            {selectedChoiceInfo.supports_live_streaming
              ? "This engine supports both file transcription and live streaming."
              : "This engine only supports file transcription."}
          </p>
        )}
      </div>

      {/* Step 2: Mode Selection */}
      <div className="space-y-2">
        <Label>2. Select Mode</Label>
        <RadioGroup
          value={mode}
          onValueChange={(val) => setMode(val as TranscriptionMode)}
          className="grid grid-cols-1 gap-2 md:grid-cols-2"
          disabled={isLiveStreaming || loading}
        >
          <div className="flex items-center space-x-2 rounded-lg border p-3">
            <RadioGroupItem value="transcription" id="mode-transcription" disabled={isLiveStreaming || loading} />
            <Label htmlFor="mode-transcription" className="cursor-pointer flex-1">
              <div className="font-medium">File Transcription</div>
              <div className="text-xs text-muted-foreground">Upload or record audio file</div>
            </Label>
          </div>
          <div className={`flex items-center space-x-2 rounded-lg border p-3 ${!supportsLiveMode ? "opacity-50" : ""}`}>
            <RadioGroupItem
              value="live"
              id="mode-live"
              disabled={isLiveStreaming || loading || !supportsLiveMode}
            />
            <Label htmlFor="mode-live" className={`flex-1 ${supportsLiveMode ? "cursor-pointer" : "cursor-not-allowed"}`}>
              <div className="font-medium flex items-center gap-2">
                Live Transcription
                {supportsLiveMode && <Radio className="h-3 w-3 text-green-500" />}
              </div>
              <div className="text-xs text-muted-foreground">
                {supportsLiveMode
                  ? "Real-time streaming via microphone"
                  : "Not supported by this engine"}
              </div>
            </Label>
          </div>
        </RadioGroup>
      </div>

      {/* Step 3: Options (common) */}
      <div className="space-y-4 pt-2 border-t">
        <Label className="text-sm font-medium">3. Options</Label>

        {/* Diarization Toggle */}
        <div className="flex items-center justify-between rounded-lg border p-3">
          <div className="space-y-0.5">
            <Label htmlFor="diarization-toggle">Enable Diarization</Label>
            <div className="text-xs text-muted-foreground">Identify different speakers</div>
          </div>
          <Switch
            id="diarization-toggle"
            checked={enableDiarization}
            onCheckedChange={setEnableDiarization}
            disabled={isLiveStreaming || loading}
          />
        </div>

        {/* Diarization Options */}
        {enableDiarization && mode === "transcription" && (
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="min-speakers">Min Speakers</Label>
              <Input
                id="min-speakers"
                type="number"
                min="1"
                value={minSpeakers}
                onChange={(e) => setMinSpeakers(e.target.value)}
                disabled={isLiveStreaming || loading}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="max-speakers">Max Speakers</Label>
              <Input
                id="max-speakers"
                type="number"
                min="1"
                value={maxSpeakers}
                onChange={(e) => setMaxSpeakers(e.target.value)}
                disabled={isLiveStreaming || loading}
              />
            </div>
          </div>
        )}

        {/* Language Selection */}
        <div className="space-y-2">
          <Label htmlFor="language">Language</Label>
          <Select
            value={language || "auto"}
            onValueChange={(val) => setLanguage(val === "auto" ? "" : val)}
            disabled={isLiveStreaming || loading}
          >
            <SelectTrigger id="language">
              <SelectValue placeholder="Auto-detect" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto-detect</SelectItem>
              <SelectItem value="en">English</SelectItem>
              <SelectItem value="zh">Chinese</SelectItem>
              <SelectItem value="es">Spanish</SelectItem>
              <SelectItem value="fr">French</SelectItem>
              <SelectItem value="de">German</SelectItem>
              <SelectItem value="ja">Japanese</SelectItem>
              <SelectItem value="ko">Korean</SelectItem>
              <SelectItem value="ru">Russian</SelectItem>
              <SelectItem value="pt">Portuguese</SelectItem>
              <SelectItem value="it">Italian</SelectItem>
              <SelectItem value="nl">Dutch</SelectItem>
              <SelectItem value="ar">Arabic</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Engine-specific options */}
        {mode === "transcription" && selectedChoice.startsWith("lib:") && selectedChoice.endsWith("whisperx/whisperx") && (
          <div className="space-y-2">
            <Label htmlFor="whisperx-asr">WhisperX ASR Model</Label>
            <Select value={whisperxAsrModel} onValueChange={(val) => setWhisperxAsrModel(val)} disabled={loading}>
              <SelectTrigger id="whisperx-asr">
                <SelectValue placeholder="Select ASR model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="tiny">tiny</SelectItem>
                <SelectItem value="base">base</SelectItem>
                <SelectItem value="small">small</SelectItem>
                <SelectItem value="small.en">small.en</SelectItem>
                <SelectItem value="medium">medium</SelectItem>
                <SelectItem value="medium.en">medium.en</SelectItem>
                <SelectItem value="large-v2">large-v2</SelectItem>
                <SelectItem value="large-v3">large-v3</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}

        {/* Live mode model selection */}
        {mode === "live" && selectedChoiceInfo?.architecture !== "funasr" && (
          <div className="space-y-2">
            <Label htmlFor="live-model">Whisper Model</Label>
            <Select
              value={liveModel}
              onValueChange={setLiveModel}
              disabled={isLiveStreaming}
            >
              <SelectTrigger id="live-model">
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="tiny">tiny (fastest)</SelectItem>
                <SelectItem value="base">base</SelectItem>
                <SelectItem value="small">small</SelectItem>
                <SelectItem value="medium">medium</SelectItem>
                <SelectItem value="large-v2">large-v2</SelectItem>
                <SelectItem value="large-v3">large-v3 (best quality)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
        {mode === "live" && selectedChoiceInfo?.architecture === "funasr" && (
          <div className="space-y-2">
            <Label htmlFor="live-model">FunASR Model</Label>
            <Select
              value={liveModel}
              onValueChange={setLiveModel}
              disabled={isLiveStreaming}
            >
              <SelectTrigger id="live-model">
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="FunAudioLLM/SenseVoiceSmall">SenseVoice (multi-language)</SelectItem>
                <SelectItem value="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch">Paraformer-zh (Mandarin)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      </div>

      {/* Step 4: Audio Input */}
      <div className="space-y-4 pt-2 border-t">
        <Label className="text-sm font-medium">4. {mode === "live" ? "Start Streaming" : "Provide Audio"}</Label>

        {mode === "transcription" ? (
          <div className="space-y-4">
            <AudioRecorderControls
              isRecording={isRecording}
              onStart={startRecording}
              onStop={stopRecording}
              onClear={recordedAudioUrl || audioFile ? clearRecording : undefined}
              supported={isRecordingSupported}
              disabled={loading}
            />

            <AudioUpload
              id="audio-file"
              label="Audio File"
              onChange={handleFileChange}
              disabled={loading || isRecording}
              fileName={audioFile ? audioFile.name : undefined}
              accept="audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm"
            />

            {audioPreviewUrl && !isRecording && (
              <AudioPlayer src={audioPreviewUrl} label="Audio Preview" />
            )}

            {isRecording && (
              <div className="flex items-center gap-2 p-4 bg-destructive/10 text-destructive rounded-lg">
                <div className="h-3 w-3 bg-destructive rounded-full animate-pulse" />
                <span className="text-sm font-medium">Recording in progress...</span>
              </div>
            )}
          </div>
        ) : (
          <>
            {/* Live mode status */}
            <div className="flex items-center justify-between">
              <Label>Status</Label>
              {getLiveStatusBadge()}
            </div>

            {!isRecordingSupported && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Not Supported</AlertTitle>
                <AlertDescription>
                  Your browser does not support audio recording. Please use a modern browser with HTTPS.
                </AlertDescription>
              </Alert>
            )}

            {liveError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{liveError}</AlertDescription>
              </Alert>
            )}

            {liveConnectionStatus === "connecting" && (
              <div className="flex items-center gap-2 p-4 bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200 rounded-lg">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm font-medium">Connecting to server...</span>
              </div>
            )}

            {liveConnectionStatus === "connected" && (
              <div className="flex items-center gap-2 p-4 bg-destructive/10 text-destructive rounded-lg">
                <div className="h-3 w-3 bg-destructive rounded-full animate-pulse" />
                <span className="text-sm font-medium">Listening...</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );

  // Output content - different for each mode
  const outputContent = mode === "transcription" ? (
    <>
      {loading && (
        <div className="space-y-4">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-32 w-full" />
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {result && !error && (
        <div className="space-y-4">
          {result.text && (
            <div className="space-y-2">
              <Label>Transcription</Label>
              <Textarea value={result.text} readOnly className="min-h-[200px] font-mono text-sm" />
            </div>
          )}

          {result.segments && result.segments.length > 0 && (
            <div className="space-y-2">
              <Label>Speaker Segments ({result.num_speakers || 0} speakers detected)</Label>
              <div className="border rounded-lg max-h-[400px] overflow-y-auto">
                <div className="divide-y">
                  {result.segments.map((segment, idx) => (
                    <div key={idx} className="p-3 hover:bg-muted/50">
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-semibold text-sm text-primary">{segment.speaker}</span>
                        <span className="text-xs text-muted-foreground whitespace-nowrap">
                          {segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
            <div>
              <span className="font-semibold">Model:</span> {result.model}
            </div>
            {result.language && (
              <div>
                <span className="font-semibold">Language:</span> {result.language}
              </div>
            )}
            {result.num_speakers !== undefined && (
              <div>
                <span className="font-semibold">Speakers:</span> {result.num_speakers}
              </div>
            )}
            <div>
              <span className="font-semibold">Processing Time:</span> {result.processing_time_ms}ms
            </div>
          </div>
        </div>
      )}

      {!loading && !error && !result && (
        <p className="text-muted-foreground text-center py-8">
          Upload or record audio, then run transcription to view results here.
        </p>
      )}
    </>
  ) : (
    <>
      <ScrollArea className="h-[500px] w-full rounded-md border p-4">
        {liveLines.length === 0 && !liveBuffer && (
          <p className="text-muted-foreground text-center py-8">
            {isLiveStreaming
              ? "Speak into your microphone..."
              : "Start live transcription to see results here."}
          </p>
        )}

        <div className="space-y-2">
          {liveLines.map((segment, idx) => (
            <div key={idx} className="p-2 rounded bg-muted/50">
              {segment.speaker > 0 && (
                <Badge variant="secondary" className="mr-2 text-xs">
                  Speaker {segment.speaker}
                </Badge>
              )}
              <span>{segment.text}</span>
              {segment.start && segment.end && (
                <span className="text-xs text-muted-foreground ml-2">
                  ({segment.start} - {segment.end})
                </span>
              )}
              {segment.detected_language && (
                <Badge variant="outline" className="ml-2 text-xs">
                  {segment.detected_language}
                </Badge>
              )}
            </div>
          ))}
        </div>

        {liveBuffer && (
          <div className="p-2 rounded bg-yellow-100/50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 mt-2">
            <span className="text-muted-foreground italic">{liveBuffer}</span>
          </div>
        )}
      </ScrollArea>

      {(liveLines.length > 0 || liveBuffer) && (
        <div className="mt-4 flex gap-4 text-sm text-muted-foreground">
          {liveStatus && <span>Status: {liveStatus}</span>}
          <span>Segments: {liveLines.length}</span>
          <span>
            Characters: {liveLines.reduce((acc, s) => acc + (s.text?.length || 0), 0) + liveBuffer.length}
          </span>
        </div>
      )}
    </>
  );

  // Footer button - different for each mode
  const footerButton = mode === "transcription" ? (
    <Button
      onClick={handleTranscribe}
      disabled={!hasAudioInput || !selectedChoice || loading}
      className="w-full"
    >
      {loading ? (
        <>
          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          {enableDiarization ? "Processing..." : "Transcribing..."}
        </>
      ) : (
        <>
          <Volume2 className="h-4 w-4 mr-2" />
          {enableDiarization ? "Transcribe & Analyze Speakers" : "Transcribe Audio"}
        </>
      )}
    </Button>
  ) : (
    <Button
      onClick={isLiveStreaming ? stopLiveStreaming : startLiveStreaming}
      disabled={!isRecordingSupported || liveConnectionStatus === "connecting" || !supportsLiveMode}
      className="w-full"
      variant={isLiveStreaming ? "destructive" : "default"}
    >
      {liveConnectionStatus === "connecting" ? (
        <>
          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          Connecting...
        </>
      ) : isLiveStreaming ? (
        <>
          <MicOff className="h-4 w-4 mr-2" />
          Stop Transcription
        </>
      ) : (
        <>
          <Mic className="h-4 w-4 mr-2" />
          Start Live Transcription
        </>
      )}
    </Button>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Volume2 className="h-8 w-8" />
          Automatic Speech Recognition
        </h1>
        <p className="text-muted-foreground">
          Transcribe speech to text with optional speaker identification
        </p>
      </div>

      <TryLayout
        input={{
          title: "Configuration",
          description: "Configure transcription settings and provide audio",
          children: inputPanel,
          footer: footerButton,
          rawPayload: rawRequest ? {
            label: "View Raw Request",
            payload: rawRequest,
          } : undefined,
        }}
        output={{
          title: "Results",
          description: mode === "live" ? "Real-time transcription" : (enableDiarization ? "Transcription with speaker analysis" : "Transcription result"),
          children: outputContent,
          rawPayload: mode === "transcription" ? {
            label: "View Raw Response",
            payload: rawResponsePayload,
          } : undefined,
        }}
      />
    </div>
  );
}

export default function AutomaticSpeechRecognitionPage() {
  return (
    <Suspense fallback={<div className="container mx-auto px-4 py-8"><div>Loading...</div></div>}>
      <AutomaticSpeechRecognitionContent />
    </Suspense>
  );
}
