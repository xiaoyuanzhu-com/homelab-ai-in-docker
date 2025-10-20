"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { TextEmbeddingInputOutput } from "@/components/text-embedding-input-output";
import { EmbeddingHistory } from "@/components/embedding-history";

interface EmbeddingResult {
  request_id: string;
  embeddings: number[][];
  dimensions: number;
  model_used: string;
  processing_time_ms: number;
}

interface EmbeddingModel {
  id: string;
  name: string;
  team: string;
  status: string;
  dimensions: number;
}

export default function EmbeddingPage() {
  const [activeTab, setActiveTab] = useState("try");

  const [texts, setTexts] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EmbeddingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");
  const [availableModels, setAvailableModels] = useState<EmbeddingModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);

  useEffect(() => {
    // Infer API base URL from current window location
    if (typeof window !== "undefined") {
      setApiBaseUrl(window.location.origin);
    }
  }, []);

  useEffect(() => {
    // Load available models
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/models");
        if (!response.ok) throw new Error("Failed to fetch models");
        const data = await response.json();
        // Filter for embedding type and downloaded models only
        const downloadedModels = data.models.filter(
          (m: any) => m.type === "embedding" && m.status === "downloaded"
        );
        setAvailableModels(downloadedModels);
        // Select first downloaded model by default
        if (downloadedModels.length > 0 && !selectedModel) {
          setSelectedModel(downloadedModels[0].id);
        }
      } catch (err) {
        console.error("Failed to fetch models:", err);
      } finally {
        setModelsLoading(false);
      }
    };
    fetchModels();
  }, [selectedModel]);

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
  };

  const handleEmbed = async () => {
    const textList = texts.split("\n").filter((t) => t.trim());
    if (textList.length === 0) return;

    if (!selectedModel) {
      toast.error("Please select a model first");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/text-to-embedding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: textList, model: selectedModel }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      toast.success("Embeddings generated successfully!", {
        description: `Processed ${data.embeddings.length} text(s) in ${data.processing_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to generate embeddings";
      setError(errorMsg);
      toast.error("Embedding generation failed", {
        description: errorMsg,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Feature Extraction</h1>
        <p className="text-muted-foreground">
          Convert text into vector representations for semantic search and similarity matching
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="mb-6">
          <TabsTrigger value="try">Try</TabsTrigger>
          <TabsTrigger value="doc">Doc</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Try Tab */}
        <TabsContent value="try">
          <TextEmbeddingInputOutput
            mode="try"
            texts={texts}
            onTextsChange={setTexts}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            availableModels={availableModels}
            modelsLoading={modelsLoading}
            result={result}
            loading={loading}
            error={error}
            onSend={handleEmbed}
          />
        </TabsContent>

        {/* API Tab */}
        <TabsContent value="doc">
          <Card>
            <CardHeader>
              <CardTitle>API Reference</CardTitle>
              <CardDescription>HTTP endpoint details and examples</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">Endpoint</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <code className="text-sm">POST /api/text-to-embedding</code>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Request Body</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "texts": [
    "The quick brown fox",
    "jumps over the lazy dog"
  ],
  "model": "all-MiniLM-L6-v2"
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Parameters</h3>
                <ul className="space-y-2 text-sm">
                  <li><code className="bg-muted px-2 py-1 rounded">texts</code> (array[string], required) - List of texts to embed</li>
                  <li><code className="bg-muted px-2 py-1 rounded">model</code> (string, optional) - Model name (default: all-MiniLM-L6-v2)</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Response</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "request_id": "uuid",
  "embeddings": [
    [0.1234, -0.5678, ...],
    [0.9012, -0.3456, ...]
  ],
  "dimensions": 384,
  "model_used": "all-MiniLM-L6-v2",
  "processing_time_ms": 45
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">cURL Example</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`curl -X POST ${apiBaseUrl}/api/text-to-embedding \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": ["Hello world", "Semantic search"]
  }'`}
                </pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history">
          <EmbeddingHistory availableModels={availableModels} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
