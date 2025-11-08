"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Volume2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle } from "lucide-react";
import { TryLayout, ModelSelector } from "@/components/try";
import { AudioUpload, AudioPlayer, AudioRecorderControls } from "@/components/inputs";

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

function AutomaticSpeechRecognitionContent() {
  const searchParams = useSearchParams();
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedChoice, setSelectedChoice] = useState<string>("");
  const [choices, setChoices] = useState<ChoiceInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [language, setLanguage] = useState<string>("");
  const [outputFormat, setOutputFormat] = useState<"transcription" | "diarization">("transcription");
  // WhisperX-specific state
  const [whisperxAsrModel, setWhisperxAsrModel] = useState<string>("large-v3");

  // Diarization-specific state
  const [minSpeakers, setMinSpeakers] = useState<string>("");
  const [maxSpeakers, setMaxSpeakers] = useState<string>("");

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [isRecordingSupported, setIsRecordingSupported] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    // Check if recording is supported
    if (typeof window !== "undefined") {
      const supported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
      setIsRecordingSupported(supported);
    }

    // Fetch available choices (models + libs) for ASR
    const fetchChoices = async () => {
      try {
        const response = await fetch("/api/task-options?task=automatic-speech-recognition");
        if (!response.ok) throw new Error("Failed to fetch ASR options");
        const data = await response.json();
        const merged: ChoiceInfo[] = (data.options || [])
          .filter((o: any) => o.status === "ready")
          .map((o: any) => ({ id: o.id, label: o.label, provider: o.provider, status: o.status, type: o.type as ChoiceType }));
        setChoices(merged);

        // Preselect logic: prefer explicit model/lib URL params, then legacy skill param, then first available
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

  const startRecording = async () => {
    // Check if mediaDevices is supported
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

        // Convert blob to file
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        setAudioFile(file);

        // Stop all tracks
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
      // Check if file is audio
      if (!file.type.startsWith("audio/") && !file.name.match(/\.(mp3|mp4|mpeg|mpga|m4a|wav|webm)$/i)) {
        toast.error("Please upload an audio file (mp3, mp4, wav, webm, etc.)");
        return;
      }

      // Clear any existing recording
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
      // Convert audio file to base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64Data = (reader.result as string).split(",")[1];

        try {
          const isModel = selectedChoice.startsWith("model:");
          const isLib = selectedChoice.startsWith("lib:");
          const id = selectedChoice.split(":", 2)[1];

          let response: Response;
          if (isModel) {
            const requestBody: ASRRequestBody = {
              audio: base64Data,
              model: id,
              output_format: outputFormat,
            };
            if (outputFormat === "transcription") {
              requestBody.language = language || undefined;
              requestBody.return_timestamps = false;
            }
            if (outputFormat === "diarization") {
              if (minSpeakers) requestBody.min_speakers = parseInt(minSpeakers);
              if (maxSpeakers) requestBody.max_speakers = parseInt(maxSpeakers);
            }
            response = await fetch("/api/automatic-speech-recognition", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(requestBody),
            });
          } else if (isLib && id === "whisperx/whisperx") {
            // WhisperX library path
            const requestBody = {
              audio: base64Data,
              asr_model: whisperxAsrModel,
              language: language || undefined,
              diarize: outputFormat === "diarization",
            };
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
          // For WhisperX, compute number of speakers for display if diarization
          if (selectedChoice.startsWith("lib:") && id === "whisperx/whisperx") {
            const speakers = new Set<string>();
            if (Array.isArray(data.segments)) {
              data.segments.forEach((s: any) => { if (s.speaker) speakers.add(s.speaker); });
            }
            setResult({ ...data, num_speakers: speakers.size || undefined });
          } else {
            setResult(data);
          }
          const successMsg = outputFormat === "diarization" ? "Speaker analysis completed!" : "Transcription completed!";
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

  const modelOptions = choices.map((c) => ({
    value: `${c.type}:${c.id}`,
    label: `${c.label} (${c.provider}) [${c.type}]`,
  }));

  const rawRequestPayload = {
    engine: selectedChoice || null,
    output_format: outputFormat,
    language: language || "auto",
    min_speakers: minSpeakers || null,
    max_speakers: maxSpeakers || null,
    audio_source: recordedAudioUrl
      ? "recorded"
      : audioFile
      ? audioFile.name
      : null,
  };

  const rawResponsePayload = result;
  const hasAudioInput = Boolean(audioFile || recordedBlob);

  const inputPanel = (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="model">Engine</Label>
        <ModelSelector
          value={selectedChoice}
          onChange={setSelectedChoice}
          options={modelOptions}
          loading={modelsLoading}
          disabled={loading}
          emptyMessage="No ASR engines ready. Visit Models/Libs to set them up."
        />
      </div>

      <div className="space-y-2">
        <Label>Mode</Label>
        <RadioGroup
          value={outputFormat}
          onValueChange={(val) => setOutputFormat(val as "transcription" | "diarization")}
          className="grid grid-cols-1 gap-2 md:grid-cols-2"
        >
          <div className="flex items-center space-x-2 rounded-lg border p-3">
            <RadioGroupItem value="transcription" id="mode-transcription" />
            <Label htmlFor="mode-transcription" className="cursor-pointer">
              Transcription
            </Label>
          </div>
          <div className="flex items-center space-x-2 rounded-lg border p-3">
            <RadioGroupItem value="diarization" id="mode-diarization" />
            <Label htmlFor="mode-diarization" className="cursor-pointer">
              Speaker Diarization
            </Label>
          </div>
        </RadioGroup>
      </div>

      {outputFormat === "transcription" && (
        <div className="space-y-2">
          <Label htmlFor="language">Language (Optional)</Label>
          <Select value={language || "auto"} onValueChange={(val) => setLanguage(val === "auto" ? "" : val)}>
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
            </SelectContent>
          </Select>
        </div>
      )}

      {outputFormat === "diarization" && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="min-speakers">Min Speakers (Optional)</Label>
            <Input
              id="min-speakers"
              type="number"
              min="1"
              placeholder="Auto"
              value={minSpeakers}
              onChange={(e) => setMinSpeakers(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="max-speakers">Max Speakers (Optional)</Label>
            <Input
              id="max-speakers"
              type="number"
              min="1"
              placeholder="Auto"
              value={maxSpeakers}
              onChange={(e) => setMaxSpeakers(e.target.value)}
            />
          </div>
        </div>
      )}

      {/* WhisperX-specific ASR model selection */}
      {selectedChoice.startsWith("lib:") && selectedChoice.endsWith("whisperx/whisperx") && (
        <div className="space-y-2">
          <Label htmlFor="whisperx-asr">WhisperX ASR Model</Label>
          <Select value={whisperxAsrModel} onValueChange={(val) => setWhisperxAsrModel(val)}>
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
    </div>
  );

  const outputContent = (
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
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Volume2 className="h-8 w-8" />
          Automatic Speech Recognition
        </h1>
        <p className="text-muted-foreground">
          Transcribe speech or identify speakers in audio files using Whisper and pyannote models
        </p>
      </div>

      <TryLayout
        input={{
          title: "Audio Processing",
          description: "Configure transcription and provide audio",
          children: inputPanel,
          footer: (
            <Button
              onClick={handleTranscribe}
              disabled={!hasAudioInput || !selectedChoice || loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  {outputFormat === "diarization" ? "Processing..." : "Transcribing..."}
                </>
              ) : (
                <>
                  <Volume2 className="h-4 w-4 mr-2" />
                  {outputFormat === "diarization" ? "Analyze Speakers" : "Transcribe Audio"}
                </>
              )}
            </Button>
          ),
          rawPayload: {
            label: "View Raw Request",
            payload: rawRequestPayload,
          },
        }}
        output={{
          title: "Results",
          description: outputFormat === "diarization" ? "Speaker analysis" : "Transcription result",
          children: outputContent,
          rawPayload: {
            label: "View Raw Response",
            payload: rawResponsePayload,
          },
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
