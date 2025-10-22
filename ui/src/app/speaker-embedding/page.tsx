"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Volume2, Upload, Loader2 } from "lucide-react";
import { toast } from "sonner";

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
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [metric, setMetric] = useState<string>("cosine");

  useEffect(() => {
    // Fetch available models
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/models?task=speaker-embedding");
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

        <TabsContent value="compare" className="space-y-6 mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Speaker Comparison</CardTitle>
              <CardDescription>
                Compare two audio files to determine if they contain the same speaker
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

              {/* Metric Selection */}
              <div className="space-y-2">
                <Label htmlFor="metric">Distance Metric</Label>
                <Select value={metric} onValueChange={setMetric}>
                  <SelectTrigger id="metric">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cosine">Cosine (Recommended)</SelectItem>
                    <SelectItem value="euclidean">Euclidean</SelectItem>
                    <SelectItem value="cityblock">Manhattan (Cityblock)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Audio File 1 */}
              <div className="space-y-2">
                <Label htmlFor="audio1">First Speaker Audio</Label>
                <input
                  id="audio1"
                  type="file"
                  accept="audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm"
                  onChange={(e) => handleFileChange(e, setAudioFile1)}
                  className="hidden"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => document.getElementById("audio1")?.click()}
                  className="w-full"
                  disabled={loading}
                >
                  <Upload className="h-4 w-4 mr-2" />
                  {audioFile1 ? audioFile1.name : "Choose First Audio File"}
                </Button>
              </div>

              {/* Audio File 2 */}
              <div className="space-y-2">
                <Label htmlFor="audio2">Second Speaker Audio</Label>
                <input
                  id="audio2"
                  type="file"
                  accept="audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm"
                  onChange={(e) => handleFileChange(e, setAudioFile2)}
                  className="hidden"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => document.getElementById("audio2")?.click()}
                  className="w-full"
                  disabled={loading}
                >
                  <Upload className="h-4 w-4 mr-2" />
                  {audioFile2 ? audioFile2.name : "Choose Second Audio File"}
                </Button>
              </div>

              {/* Compare Button */}
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

              {/* Error Display */}
              {error && (
                <div className="p-4 bg-destructive/10 text-destructive rounded-lg text-sm">
                  {error}
                </div>
              )}

              {/* Comparison Result */}
              {comparisonResult && (
                <div className="space-y-4 mt-6">
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
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="extract" className="space-y-6 mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Extract Embedding</CardTitle>
              <CardDescription>
                Extract speaker embedding vector from an audio file
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Model Selection */}
              <div className="space-y-2">
                <Label htmlFor="extract-model">Model</Label>
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
                    <SelectTrigger id="extract-model">
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

              {/* Audio File */}
              <div className="space-y-2">
                <Label htmlFor="single-audio">Audio File</Label>
                <input
                  id="single-audio"
                  type="file"
                  accept="audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm"
                  onChange={(e) => handleFileChange(e, setSingleAudioFile)}
                  className="hidden"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => document.getElementById("single-audio")?.click()}
                  className="w-full"
                  disabled={loading}
                >
                  <Upload className="h-4 w-4 mr-2" />
                  {singleAudioFile ? singleAudioFile.name : "Choose Audio File"}
                </Button>
              </div>

              {/* Extract Button */}
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

              {/* Error Display */}
              {error && (
                <div className="p-4 bg-destructive/10 text-destructive rounded-lg text-sm">
                  {error}
                </div>
              )}

              {/* Embedding Result */}
              {embeddingResult && (
                <div className="space-y-4 mt-6">
                  <div className="space-y-2">
                    <Label>Embedding Vector</Label>
                    <div className="p-4 bg-muted rounded-lg">
                      <div className="text-xs font-mono break-all max-h-40 overflow-y-auto">
                        [{embeddingResult.embedding.map(v => v.toFixed(4)).join(", ")}]
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                    <div>
                      <span className="font-semibold">Dimension:</span> {embeddingResult.dimension}
                    </div>
                    <div>
                      <span className="font-semibold">Model:</span> {embeddingResult.model}
                    </div>
                    <div>
                      <span className="font-semibold">Processing Time:</span> {embeddingResult.processing_time_ms}ms
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
                Speaker embedding models for verification and identification
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Please visit the <a href="/models" className="text-primary hover:underline">Models page</a> to download speaker embedding models.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
