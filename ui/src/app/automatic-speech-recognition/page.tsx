"use client";

import { useState, useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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

interface SkillInfo {
  id: string;
  label: string;
  provider: string;
  tasks: string[];
  size_mb: number;
  parameters_m: number;
  gpu_memory_mb: number;
  reference_url: string;
  status: string;
  downloaded_size_mb?: number;
  error_message?: string;
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

export default function AutomaticSpeechRecognitionPage() {
  const searchParams = useSearchParams();
  const [activeTab, setActiveTab] = useState("try");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<SkillInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [language, setLanguage] = useState<string>("");
  const [outputFormat, setOutputFormat] = useState<"transcription" | "diarization">("transcription");

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

    // Fetch available skills
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/skills?task=automatic-speech-recognition");
        if (!response.ok) {
          throw new Error("Failed to fetch skills");
        }
        const data = await response.json();
        // Filter for ready skills only in Try tab
        const downloadedModels = data.skills.filter(
          (s: SkillInfo) => s.status === "ready"
        );
        setAvailableModels(downloadedModels);

        // Check if skill query param is provided
        const skillParam = searchParams.get("skill");
        if (skillParam && downloadedModels.some((s: SkillInfo) => s.id === skillParam)) {
          // Pre-select the skill from query param if it exists and is ready
          setSelectedModel(skillParam);
        } else if (downloadedModels.length > 0) {
          // Set first downloaded model as default
          setSelectedModel(downloadedModels[0].id);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        toast.error("Failed to load available skills");
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
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
    if (!audioFile || !selectedModel) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Convert audio file to base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64Data = (reader.result as string).split(",")[1];

        try {
          const requestBody: ASRRequestBody = {
            audio: base64Data,
            model: selectedModel,
            output_format: outputFormat,
          };

          // Add transcription-specific params
          if (outputFormat === "transcription") {
            requestBody.language = language || undefined;
            requestBody.return_timestamps = false;
          }

          // Add diarization-specific params
          if (outputFormat === "diarization") {
            if (minSpeakers) requestBody.min_speakers = parseInt(minSpeakers);
            if (maxSpeakers) requestBody.max_speakers = parseInt(maxSpeakers);
          }

          const response = await fetch("/api/automatic-speech-recognition", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          setResult(data);
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

  const modelOptions = availableModels.map((skill) => ({
    value: skill.id,
    label: `${skill.label} (${skill.parameters_m}M params)`,
  }));

  const rawRequestPayload = {
    model: selectedModel || null,
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
        <Label htmlFor="model">Model</Label>
        <ModelSelector
          value={selectedModel}
          onChange={setSelectedModel}
          options={modelOptions}
          loading={modelsLoading}
          disabled={loading}
          emptyMessage="No ASR skills downloaded. Visit the Skills page to install one."
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
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Volume2 className="h-8 w-8" />
          Automatic Speech Recognition
        </h1>
        <p className="text-muted-foreground">
          Transcribe speech or identify speakers in audio files using Whisper and pyannote models
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="try">Try It</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
        </TabsList>

        <TabsContent value="try" className="mt-6">
          <TryLayout
            input={{
              title: "Audio Processing",
              description: "Configure transcription and provide audio",
              children: inputPanel,
              footer: (
                <Button
                  onClick={handleTranscribe}
                  disabled={!hasAudioInput || !selectedModel || loading}
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
        </TabsContent>

        <TabsContent value="models" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Available Models</CardTitle>
              <CardDescription>
                Whisper models for automatic speech recognition
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Please visit the <a href="/models" className="text-primary hover:underline">Skills page</a> to download ASR models.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
