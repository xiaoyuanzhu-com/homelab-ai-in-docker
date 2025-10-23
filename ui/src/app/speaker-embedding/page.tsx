"use client";

import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Volume2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { TryLayout, ModelSelector } from "@/components/try";
import { AudioUpload } from "@/components/inputs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface EmbeddingResult {
  request_id: string;
  embedding: number[];
  dimension: number;
  model: string;
  duration?: number;
  processing_time_ms: number;
}

interface ComparisonResult {
  request_id: string;
  distance: number;
  similarity: number;
  metric: string;
  model: string;
  processing_time_ms: number;
}

interface SkillInfo {
  id: string;
  label: string;
  provider: string;
  tasks: string[];

  size_mb: number;
  parameters_m: number;
  gpu_memory_mb: number;
  link: string;
  status: string;
  downloaded_size_mb?: number;
  error_message?: string;
}

export default function SpeakerEmbeddingPage() {
  const [activeTab, setActiveTab] = useState("compare");
  const [audioFile1, setAudioFile1] = useState<File | null>(null);
  const [audioFile2, setAudioFile2] = useState<File | null>(null);
  const [singleAudioFile, setSingleAudioFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);
  const [embeddingResult, setEmbeddingResult] = useState<EmbeddingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("pyannote/embedding");
  const [availableModels, setAvailableModels] = useState<SkillInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [metric, setMetric] = useState<string>("cosine");

  useEffect(() => {
    // Fetch available skills
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/skills?task=speaker-embedding");
        if (!response.ok) {
          throw new Error("Failed to fetch skills");
        }
        const data = await response.json();
        // Filter for downloaded skills only in Try tab
        const downloadedModels = data.skills.filter(
          (s: SkillInfo) => m.status === "downloaded"
        );
        setAvailableModels(downloadedModels);
        // Set first downloaded model as default
        if (downloadedModels.length > 0) {
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
  }, []);

  const handleFileChange = (
    e: React.ChangeEvent<HTMLInputElement>,
    setter: (file: File | null) => void
  ) => {
    const file = e.target.files?.[0];
    if (file) {
      // Check if file is audio
      if (!file.type.startsWith("audio/") && !file.name.match(/\.(mp3|mp4|mpeg|mpga|m4a|wav|webm)$/i)) {
        toast.error("Please upload an audio file (mp3, mp4, wav, webm, etc.)");
        return;
      }
      setter(file);
      setError(null);
      setComparisonResult(null);
      setEmbeddingResult(null);
    }
  };

  const handleCompare = async () => {
    if (!audioFile1 || !audioFile2 || !selectedModel) return;

    setLoading(true);
    setError(null);
    setComparisonResult(null);

    try {
      // Convert both audio files to base64
      const reader1 = new FileReader();
      const reader2 = new FileReader();

      const base64Audio1 = await new Promise<string>((resolve) => {
        reader1.onloadend = () => resolve((reader1.result as string).split(",")[1]);
        reader1.readAsDataURL(audioFile1);
      });

      const base64Audio2 = await new Promise<string>((resolve) => {
        reader2.onloadend = () => resolve((reader2.result as string).split(",")[1]);
        reader2.readAsDataURL(audioFile2);
      });

      const response = await fetch("/api/speaker-embedding/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio1: base64Audio1,
          audio2: base64Audio2,
          model: selectedModel,
          metric: metric,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setComparisonResult(data);
      toast.success("Comparison completed!", {
        description: `Processed in ${data.processing_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to compare speakers";
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleExtract = async () => {
    if (!singleAudioFile || !selectedModel) return;

    setLoading(true);
    setError(null);
    setEmbeddingResult(null);

    try {
      // Convert audio file to base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64Data = (reader.result as string).split(",")[1];

        try {
          const response = await fetch("/api/speaker-embedding/extract", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              audio: base64Data,
              model: selectedModel,
              mode: "whole",
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          setEmbeddingResult(data);
          toast.success("Embedding extracted!", {
            description: `Processed in ${data.processing_time_ms}ms`,
          });
        } catch (err) {
          const errorMsg = err instanceof Error ? err.message : "Failed to extract embedding";
          setError(errorMsg);
          toast.error(errorMsg);
        } finally {
          setLoading(false);
        }
      };
      reader.readAsDataURL(singleAudioFile);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to process audio file";
      setError(errorMsg);
      toast.error(errorMsg);
      setLoading(false);
    }
  };

  const modelOptions = availableModels.map((skill) => ({
    value: model.id,
    label: model.parameters_m ? `${skill.label} (${model.parameters_m}M params)` : skill.label,
  }));

  const formatFileSummary = (file: File | null) => {
    if (!file) return null;
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    return `${file.name} (${sizeKb}KB)`;
  };

  const compareRequestPayload = {
    model: selectedModel || null,
    metric,
    audio1: formatFileSummary(audioFile1),
    audio2: formatFileSummary(audioFile2),
  };

  const compareResponsePayload = comparisonResult;

  const extractRequestPayload = {
    model: selectedModel || null,
    audio: formatFileSummary(singleAudioFile),
  };

  const extractResponsePayload = embeddingResult;

  const compareInput = (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="compare-model">Model</Label>
        <ModelSelector
          value={selectedModel}
          onChange={setSelectedModel}
          options={modelOptions}
          loading={modelsLoading}
          disabled={loading}
          placeholder="Select a model"
          emptyMessage="No speaker embedding skills downloaded. Visit the Models tab to install one."
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="metric">Distance Metric</Label>
        <Select value={metric} onValueChange={setMetric} disabled={loading}>
          <SelectTrigger id="metric">
            <SelectValue placeholder="Select metric" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cosine">Cosine (Recommended)</SelectItem>
            <SelectItem value="euclidean">Euclidean</SelectItem>
            <SelectItem value="cityblock">Manhattan (Cityblock)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <AudioUpload
        id="compare-audio-1"
        label="First Speaker Audio"
        onChange={(e) => handleFileChange(e, setAudioFile1)}
        disabled={loading}
        fileName={audioFile1?.name}
        helperText="Supports mp3, mp4, wav, webm, m4a."
        accept="audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm"
      />

      <AudioUpload
        id="compare-audio-2"
        label="Second Speaker Audio"
        onChange={(e) => handleFileChange(e, setAudioFile2)}
        disabled={loading}
        fileName={audioFile2?.name}
        helperText="Supports mp3, mp4, wav, webm, m4a."
        accept="audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm"
      />
    </div>
  );

  const comparisonInsight = comparisonResult ? (
    <div className="space-y-4">
      <div className="p-6 bg-muted rounded-lg">
        <div className="text-center space-y-4">
          <div>
            <div className="text-sm text-muted-foreground">Similarity Score</div>
            <div className="text-4xl font-bold text-primary">
              {(comparisonResult.similarity * 100).toFixed(1)}%
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-muted-foreground">Distance</div>
              <div className="font-semibold">{comparisonResult.distance.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Metric</div>
              <div className="font-semibold capitalize">{comparisonResult.metric}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
        <div>
          <span className="font-semibold">Model:</span> {comparisonResult.model}
        </div>
        <div>
          <span className="font-semibold">Processing Time:</span> {comparisonResult.processing_time_ms}ms
        </div>
      </div>
    </div>
  ) : null;

  const compareOutput = (
    <div className="space-y-4">
      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Comparing audio...
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {comparisonInsight}

      {!loading && !error && !comparisonResult && (
        <p className="text-muted-foreground text-sm text-center py-6">
          Provide two audio clips and run the comparison to see similarity metrics here.
        </p>
      )}
    </div>
  );

  const extractInput = (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="extract-model">Model</Label>
        <ModelSelector
          value={selectedModel}
          onChange={setSelectedModel}
          options={modelOptions}
          loading={modelsLoading}
          disabled={loading}
          placeholder="Select a model"
          emptyMessage="No speaker embedding skills downloaded. Visit the Models tab to install one."
        />
      </div>

      <AudioUpload
        id="extract-audio"
        label="Speaker Audio"
        onChange={(e) => handleFileChange(e, setSingleAudioFile)}
        disabled={loading}
        fileName={singleAudioFile?.name}
        helperText="Supports mp3, mp4, wav, webm, m4a."
        accept="audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm"
      />
    </div>
  );

  const embeddingPreview = embeddingResult?.embedding
    ? embeddingResult.embedding
        .slice(0, 16)
        .map((value) => value.toFixed(4))
        .join(", ") +
      (embeddingResult.embedding.length > 16 ? " …" : "")
    : "";

  const extractOutput = (
    <div className="space-y-4">
      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Processing audio...
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {embeddingResult && !error && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
            <div>
              <span className="font-semibold">Dimension:</span> {embeddingResult.dimension}
            </div>
            <div>
              <span className="font-semibold">Processing Time:</span> {embeddingResult.processing_time_ms}ms
            </div>
            {embeddingResult.duration !== undefined && (
              <div>
                <span className="font-semibold">Duration:</span> {embeddingResult.duration.toFixed(2)}s
              </div>
            )}
            <div>
              <span className="font-semibold">Model:</span> {embeddingResult.model}
            </div>
          </div>

          <div className="space-y-2">
            <Label>Embedding Preview</Label>
            <Textarea
              value={embeddingPreview}
              readOnly
              className="min-h-[200px] font-mono text-sm"
              placeholder="Embedding vector preview"
            />
            <p className="text-xs text-muted-foreground">
              Showing the first {Math.min(16, embeddingResult.embedding.length)} values out of {embeddingResult.embedding.length}.
            </p>
          </div>
        </div>
      )}

      {!loading && !error && !embeddingResult && (
        <p className="text-muted-foreground text-sm text-center py-6">
          Upload an audio clip and run extraction to view embedding details here.
        </p>
      )}
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Volume2 className="h-8 w-8" />
          Speaker Embedding
        </h1>
        <p className="text-muted-foreground">
          Extract speaker embeddings for verification and identification
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full max-w-md grid-cols-3">
          <TabsTrigger value="compare">Compare</TabsTrigger>
          <TabsTrigger value="extract">Extract</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
        </TabsList>

        <TabsContent value="compare" className="mt-6">
          <TryLayout
            input={{
              title: "Speaker Comparison",
              description: "Compare two audio clips to determine if they contain the same speaker",
              children: compareInput,
              footer: (
                <Button
                  onClick={handleCompare}
                  disabled={!audioFile1 || !audioFile2 || !selectedModel || loading}
                  className="w-full"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Comparing...
                    </>
                  ) : (
                    <>
                      <Volume2 className="h-4 w-4 mr-2" />
                      Compare Speakers
                    </>
                  )}
                </Button>
              ),
              rawPayload: {
                label: "View Raw Request",
                payload: compareRequestPayload,
              },
            }}
            output={{
              title: "Results",
              description: "Similarity metrics",
              children: compareOutput,
              rawPayload: {
                label: "View Raw Response",
                payload: compareResponsePayload,
              },
            }}
          />
        </TabsContent>

        <TabsContent value="extract" className="mt-6">
          <TryLayout
            input={{
              title: "Extract Embedding",
              description: "Generate a speaker embedding vector from audio",
              children: extractInput,
              footer: (
                <Button
                  onClick={handleExtract}
                  disabled={!singleAudioFile || !selectedModel || loading}
                  className="w-full"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Extracting...
                    </>
                  ) : (
                    <>
                      <Volume2 className="h-4 w-4 mr-2" />
                      Extract Embedding
                    </>
                  )}
                </Button>
              ),
              rawPayload: {
                label: "View Raw Request",
                payload: extractRequestPayload,
              },
            }}
            output={{
              title: "Embedding",
              description: "Vector details",
              children: extractOutput,
              rawPayload: {
                label: "View Raw Response",
                payload: extractResponsePayload,
              },
            }}
          />
        </TabsContent>
        <TabsContent value="models" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Available Models</CardTitle>
              <CardDescription>
                Speaker embedding models for verification and identification
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Please visit the <a href="/models" className="text-primary hover:underline">Skills page</a> to download speaker embedding models.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
