"use client";

import { useState, useEffect, Suspense, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Volume2, Loader2, Play, Download } from "lucide-react";
import { toast } from "sonner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle } from "lucide-react";
import { TryLayout, ModelSelector } from "@/components/try";
import { AudioUpload, AudioPlayer } from "@/components/inputs";
import { Slider } from "@/components/ui/slider";

interface TTSResult {
  request_id: string;
  audio: string;
  sample_rate: number;
  duration_seconds: number;
  model: string;
  mode: string;
  processing_time_ms: number;
}

interface ModelInfo {
  id: string;
  label: string;
  provider: string;
  status: string;
}

type TTSMode = "zero_shot" | "cross_lingual" | "instruct" | "sft";

function TextToSpeechContent() {
  const searchParams = useSearchParams();

  // Model state
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Input state
  const [text, setText] = useState<string>("Hello, welcome to the AI voice synthesis demo.");
  const [mode, setMode] = useState<TTSMode>("instruct");
  const [instruction, setInstruction] = useState<string>("Speak naturally with a friendly tone");
  const [promptText, setPromptText] = useState<string>("");
  const [promptAudio, setPromptAudio] = useState<string | null>(null);
  const [promptAudioFile, setPromptAudioFile] = useState<File | null>(null);
  const [speakerId, setSpeakerId] = useState<string>("中文女");
  const [speed, setSpeed] = useState<number>(1.0);

  // Result state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TTSResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rawRequest, setRawRequest] = useState<any>(null);

  // Audio playback
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Fetch models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/models?task=text-to-speech");
        if (!response.ok) throw new Error("Failed to fetch TTS models");
        const data = await response.json();
        const availableModels = (data.models || []).filter(
          (m: ModelInfo) => m.status === "ready"
        );
        setModels(availableModels);

        // Set initial model from URL or default
        const modelParam = searchParams.get("model");
        if (modelParam && availableModels.find((m: ModelInfo) => m.id === modelParam)) {
          setSelectedModel(modelParam);
        } else if (availableModels.length > 0) {
          // Prefer Fun-CosyVoice3 if available
          const cosyvoice3 = availableModels.find((m: ModelInfo) =>
            m.id.includes("CosyVoice3") || m.id.includes("Fun-CosyVoice3")
          );
          setSelectedModel(cosyvoice3?.id || availableModels[0].id);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        toast.error("Failed to load TTS models");
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
  }, [searchParams]);

  // Handle prompt audio file upload
  const handlePromptAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("audio/") && !file.name.match(/\.(mp3|wav|webm|m4a|ogg)$/i)) {
        toast.error("Please upload an audio file");
        return;
      }
      setPromptAudioFile(file);

      // Convert to base64
      const reader = new FileReader();
      reader.onloadend = () => {
        setPromptAudio(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const clearPromptAudio = () => {
    setPromptAudio(null);
    setPromptAudioFile(null);
  };

  // Generate speech
  const handleGenerate = async () => {
    if (!text.trim() || !selectedModel) return;

    // Validate mode requirements
    if ((mode === "zero_shot" || mode === "cross_lingual") && !promptAudio) {
      toast.error(`Reference audio is required for ${mode} mode`);
      return;
    }
    if (mode === "instruct" && !instruction.trim()) {
      toast.error("Instruction is required for instruct mode");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const requestBody: any = {
        text,
        model: selectedModel,
        mode,
        speed,
      };

      if (mode === "zero_shot" || mode === "cross_lingual") {
        requestBody.prompt_audio = promptAudio;
        if (promptText.trim()) {
          requestBody.prompt_text = promptText;
        }
      }

      if (mode === "instruct") {
        requestBody.instruction = instruction;
        if (promptAudio) {
          requestBody.prompt_audio = promptAudio;
        }
      }

      if (mode === "sft") {
        requestBody.speaker_id = speakerId;
      }

      setRawRequest({
        ...requestBody,
        prompt_audio: requestBody.prompt_audio ? "<base64 audio data>" : undefined,
      });

      const response = await fetch("/api/text-to-speech", {
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

      toast.success("Speech generated!", {
        description: `Duration: ${data.duration_seconds.toFixed(1)}s, Processing: ${data.processing_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to generate speech";
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  // Download audio
  const handleDownload = () => {
    if (!result?.audio) return;

    const link = document.createElement("a");
    link.href = result.audio;
    link.download = `tts_output_${Date.now()}.wav`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const modelOptions = models.map((m) => ({
    value: m.id,
    label: `${m.label} (${m.provider})`,
  }));

  // Mode descriptions
  const modeDescriptions: Record<TTSMode, string> = {
    zero_shot: "Clone a voice from reference audio (requires audio sample)",
    cross_lingual: "Synthesize in one language using a voice from another language",
    instruct: "Control speech style with natural language instructions",
    sft: "Use pre-trained speaker voices (no reference audio needed)",
  };

  // Input panel
  const inputPanel = (
    <div className="space-y-4">
      {/* Model Selection */}
      <div className="space-y-2">
        <Label htmlFor="model">Model</Label>
        <ModelSelector
          value={selectedModel}
          onChange={setSelectedModel}
          options={modelOptions}
          loading={modelsLoading}
          disabled={loading}
          emptyMessage="No TTS models ready. Visit Models to download one."
        />
      </div>

      {/* Mode Selection */}
      <div className="space-y-2">
        <Label htmlFor="mode">Synthesis Mode</Label>
        <Select
          value={mode}
          onValueChange={(val) => setMode(val as TTSMode)}
          disabled={loading}
        >
          <SelectTrigger id="mode">
            <SelectValue placeholder="Select mode" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="instruct">Instruct (Style Control)</SelectItem>
            <SelectItem value="zero_shot">Zero-shot (Voice Cloning)</SelectItem>
            <SelectItem value="cross_lingual">Cross-lingual</SelectItem>
            <SelectItem value="sft">SFT (Pre-trained Voices)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">{modeDescriptions[mode]}</p>
      </div>

      {/* Text Input */}
      <div className="space-y-2">
        <Label htmlFor="text">Text to Synthesize</Label>
        <Textarea
          id="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to convert to speech..."
          className="min-h-[100px]"
          disabled={loading}
        />
      </div>

      {/* Mode-specific options */}
      {mode === "instruct" && (
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="instruction">Instruction</Label>
            <Textarea
              id="instruction"
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              placeholder="e.g., Speak happily, Use a Sichuan dialect, Speak slowly and clearly"
              className="min-h-[60px]"
              disabled={loading}
            />
            <p className="text-xs text-muted-foreground">
              Control emotions, speed, accent, dialect, etc.
            </p>
          </div>

          <div className="space-y-2">
            <Label>Reference Audio (Required for CosyVoice2/3)</Label>
            <AudioUpload
              id="instruct-prompt-audio"
              label="Upload Voice Sample"
              onChange={handlePromptAudioChange}
              disabled={loading}
              fileName={promptAudioFile?.name}
              accept="audio/*,.mp3,.wav,.webm,.m4a,.ogg"
            />
            {promptAudio && (
              <div className="flex items-center gap-2">
                <AudioPlayer src={promptAudio} label="Reference Audio" />
                <Button variant="outline" size="sm" onClick={clearPromptAudio}>
                  Clear
                </Button>
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              CosyVoice2/3 instruct mode requires a reference audio for voice cloning.
            </p>
          </div>
        </div>
      )}

      {(mode === "zero_shot" || mode === "cross_lingual") && (
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Reference Audio (Required)</Label>
            <AudioUpload
              id="prompt-audio"
              label="Upload Voice Sample"
              onChange={handlePromptAudioChange}
              disabled={loading}
              fileName={promptAudioFile?.name}
              accept="audio/*,.mp3,.wav,.webm,.m4a,.ogg"
            />
            {promptAudio && (
              <div className="flex items-center gap-2">
                <AudioPlayer src={promptAudio} label="Reference Audio" />
                <Button variant="outline" size="sm" onClick={clearPromptAudio}>
                  Clear
                </Button>
              </div>
            )}
          </div>

          {mode === "zero_shot" && (
            <div className="space-y-2">
              <Label htmlFor="prompt-text">
                Reference Transcript {selectedModel?.includes("CosyVoice3") ? "(Required for CosyVoice3)" : "(Optional)"}
              </Label>
              <Textarea
                id="prompt-text"
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                placeholder={
                  selectedModel?.includes("CosyVoice3")
                    ? "REQUIRED: Type the exact words spoken in your reference audio"
                    : "Transcript of the reference audio (helps improve quality)"
                }
                className="min-h-[60px]"
                disabled={loading}
              />
              {selectedModel?.includes("CosyVoice3") && (
                <p className="text-xs text-muted-foreground">
                  CosyVoice3 requires the exact transcript of your reference audio. Without it, the output will be garbled.
                  If you don&apos;t have the transcript, use &quot;Cross-lingual&quot; mode instead.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {mode === "sft" && (
        <div className="space-y-2">
          <Label htmlFor="speaker">Pre-trained Speaker</Label>
          <Select
            value={speakerId}
            onValueChange={setSpeakerId}
            disabled={loading}
          >
            <SelectTrigger id="speaker">
              <SelectValue placeholder="Select speaker" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="中文女">Chinese Female</SelectItem>
              <SelectItem value="中文男">Chinese Male</SelectItem>
              <SelectItem value="英文女">English Female</SelectItem>
              <SelectItem value="英文男">English Male</SelectItem>
              <SelectItem value="日语男">Japanese Male</SelectItem>
              <SelectItem value="粤语女">Cantonese Female</SelectItem>
              <SelectItem value="韩语女">Korean Female</SelectItem>
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Speed Control */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="speed">Speed</Label>
          <span className="text-sm text-muted-foreground">{speed.toFixed(1)}x</span>
        </div>
        <Slider
          id="speed"
          value={[speed]}
          onValueChange={(val) => setSpeed(val[0])}
          min={0.5}
          max={2.0}
          step={0.1}
          disabled={loading}
        />
      </div>
    </div>
  );

  // Output content
  const outputContent = (
    <>
      {loading && (
        <div className="space-y-4">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-24 w-full" />
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
          <div className="space-y-2">
            <Label>Generated Audio</Label>
            <audio
              ref={audioRef}
              src={result.audio}
              controls
              className="w-full"
            />
          </div>

          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => audioRef.current?.play()}
            >
              <Play className="h-4 w-4 mr-2" />
              Play
            </Button>
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
          </div>

          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground pt-4 border-t">
            <div>
              <span className="font-semibold">Duration:</span> {result.duration_seconds.toFixed(2)}s
            </div>
            <div>
              <span className="font-semibold">Sample Rate:</span> {result.sample_rate} Hz
            </div>
            <div>
              <span className="font-semibold">Mode:</span> {result.mode}
            </div>
            <div>
              <span className="font-semibold">Processing:</span> {result.processing_time_ms}ms
            </div>
          </div>
        </div>
      )}

      {!loading && !error && !result && (
        <p className="text-muted-foreground text-center py-8">
          Enter text and click Generate to create speech.
        </p>
      )}
    </>
  );

  // Footer button
  const footerButton = (
    <Button
      onClick={handleGenerate}
      disabled={!text.trim() || !selectedModel || loading}
      className="w-full"
    >
      {loading ? (
        <>
          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          Generating...
        </>
      ) : (
        <>
          <Volume2 className="h-4 w-4 mr-2" />
          Generate Speech
        </>
      )}
    </Button>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Volume2 className="h-8 w-8" />
          Text to Speech
        </h1>
        <p className="text-muted-foreground">
          Synthesize natural speech from text using CosyVoice models
        </p>
      </div>

      <TryLayout
        input={{
          title: "Configuration",
          description: "Configure TTS settings and enter text",
          children: inputPanel,
          footer: footerButton,
          rawPayload: rawRequest ? {
            label: "View Raw Request",
            payload: rawRequest,
          } : undefined,
        }}
        output={{
          title: "Output",
          description: "Generated audio output",
          children: outputContent,
          rawPayload: result ? {
            label: "View Raw Response",
            payload: {
              ...result,
              audio: "<base64 audio data>",
            },
          } : undefined,
        }}
      />
    </div>
  );
}

export default function TextToSpeechPage() {
  return (
    <Suspense fallback={<div className="container mx-auto px-4 py-8"><div>Loading...</div></div>}>
      <TextToSpeechContent />
    </Suspense>
  );
}
