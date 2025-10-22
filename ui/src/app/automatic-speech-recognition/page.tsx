"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Volume2, Upload, Loader2, Mic, StopCircle, Trash2 } from "lucide-react";
import { toast } from "sonner";

interface TranscriptionResult {
  request_id: string;
  text: string;
  model: string;
  language: string | null;
  processing_time_ms: number;
}

interface ModelInfo {
  id: string;
  name: string;
  team: string;
  type: string;
  task: string;
  size_mb: number;
  parameters_m: number;
  gpu_memory_mb: number;
  link: string;
  status: string;
  downloaded_size_mb?: number;
  error_message?: string;
}

export default function AutomaticSpeechRecognitionPage() {
  const [activeTab, setActiveTab] = useState("try");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [language, setLanguage] = useState<string>("");

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [isRecordingSupported, setIsRecordingSupported] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    // Check if recording is supported
    if (typeof window !== "undefined") {
      const supported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
      setIsRecordingSupported(supported);
    }

    // Fetch available models
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/models?task=automatic-speech-recognition");
        if (!response.ok) {
          throw new Error("Failed to fetch models");
        }
        const data = await response.json();
        // Filter for downloaded models only in Try tab
        const downloadedModels = data.models.filter(
          (m: ModelInfo) => m.status === "downloaded"
        );
        setAvailableModels(downloadedModels);
        // Set first downloaded model as default
        if (downloadedModels.length > 0) {
          setSelectedModel(downloadedModels[0].id);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        toast.error("Failed to load available models");
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
  }, []);

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
          const response = await fetch("/api/automatic-speech-recognition", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              audio: base64Data,
              model: selectedModel,
              language: language || undefined,
              return_timestamps: false,
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          setResult(data);
          toast.success("Transcription completed!", {
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

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Volume2 className="h-8 w-8" />
          Automatic Speech Recognition
        </h1>
        <p className="text-muted-foreground">
          Transcribe speech from audio files to text using Whisper models
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="try">Try It</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
        </TabsList>

        <TabsContent value="try" className="space-y-6 mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Audio Transcription</CardTitle>
              <CardDescription>
                Record audio from your microphone or upload an audio file to transcribe
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Model Selection */}
              <div className="space-y-2">
                <Label htmlFor="model">Model</Label>
                {modelsLoading ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading models...
                  </div>
                ) : availableModels.length === 0 ? (
                  <div className="text-sm text-muted-foreground">
                    No models downloaded. Please download a model from the Models tab.
                  </div>
                ) : (
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger id="model">
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableModels.map((model) => (
                        <SelectItem key={model.id} value={model.id}>
                          {model.name} ({model.parameters_m}M params)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>

              {/* Language Selection (Optional) */}
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

              {/* Recording Controls */}
              <div className="space-y-4">
                <Label>Audio Input</Label>

                {/* Record from Microphone */}
                {isRecordingSupported ? (
                  <>
                    <div className="flex items-center gap-2">
                      {!isRecording ? (
                        <Button
                          type="button"
                          variant="default"
                          onClick={startRecording}
                          disabled={loading}
                          className="flex-1"
                        >
                          <Mic className="h-4 w-4 mr-2" />
                          Record from Microphone
                        </Button>
                      ) : (
                        <Button
                          type="button"
                          variant="destructive"
                          onClick={stopRecording}
                          className="flex-1"
                        >
                          <StopCircle className="h-4 w-4 mr-2" />
                          Stop Recording
                        </Button>
                      )}

                      {(recordedAudioUrl || audioFile) && !isRecording && (
                        <Button
                          type="button"
                          variant="outline"
                          size="icon"
                          onClick={clearRecording}
                          disabled={loading}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      )}
                    </div>

                    {/* Or Upload File */}
                    {!isRecording && (
                      <div className="relative">
                        <div className="absolute inset-0 flex items-center">
                          <span className="w-full border-t" />
                        </div>
                        <div className="relative flex justify-center text-xs uppercase">
                          <span className="bg-background px-2 text-muted-foreground">
                            Or upload a file
                          </span>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="p-3 bg-muted rounded-lg text-sm text-muted-foreground">
                    <p className="mb-2">
                      <strong>Note:</strong> Audio recording is not available in your current environment.
                    </p>
                    <p className="text-xs">
                      Recording requires HTTPS or localhost. Please upload an audio file instead.
                    </p>
                  </div>
                )}

                {!isRecording && (
                  <div className="flex items-center gap-2">
                    <input
                      id="audio-file"
                      type="file"
                      accept="audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => document.getElementById("audio-file")?.click()}
                      className="flex-1"
                      disabled={loading}
                    >
                      <Upload className="h-4 w-4 mr-2" />
                      {audioFile && !recordedBlob ? audioFile.name : "Choose Audio File"}
                    </Button>
                  </div>
                )}

                {/* Audio Preview */}
                {(recordedAudioUrl || audioFile) && !isRecording && (
                  <div className="space-y-2">
                    <Label>Audio Preview</Label>
                    <audio
                      controls
                      className="w-full"
                      src={recordedAudioUrl || (audioFile ? URL.createObjectURL(audioFile) : undefined)}
                    />
                    {recordedBlob && (
                      <p className="text-xs text-muted-foreground">
                        Recorded audio (webm format)
                      </p>
                    )}
                  </div>
                )}

                {/* Recording Indicator */}
                {isRecording && (
                  <div className="flex items-center gap-2 p-4 bg-destructive/10 text-destructive rounded-lg">
                    <div className="h-3 w-3 bg-destructive rounded-full animate-pulse" />
                    <span className="text-sm font-medium">Recording in progress...</span>
                  </div>
                )}
              </div>

              {/* Transcribe Button */}
              <Button
                onClick={handleTranscribe}
                disabled={!audioFile || !selectedModel || loading}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Transcribing...
                  </>
                ) : (
                  <>
                    <Volume2 className="h-4 w-4 mr-2" />
                    Transcribe Audio
                  </>
                )}
              </Button>

              {/* Error Display */}
              {error && (
                <div className="p-4 bg-destructive/10 text-destructive rounded-lg text-sm">
                  {error}
                </div>
              )}

              {/* Result Display */}
              {result && (
                <div className="space-y-4 mt-6">
                  <div className="space-y-2">
                    <Label>Transcription</Label>
                    <Textarea
                      value={result.text}
                      readOnly
                      className="min-h-[200px] font-mono text-sm"
                    />
                  </div>

                  <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                    <div>
                      <span className="font-semibold">Model:</span> {result.model}
                    </div>
                    {result.language && (
                      <div>
                        <span className="font-semibold">Language:</span> {result.language}
                      </div>
                    )}
                    <div>
                      <span className="font-semibold">Processing Time:</span> {result.processing_time_ms}ms
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
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
                Please visit the <a href="/models" className="text-primary hover:underline">Models page</a> to download ASR models.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
