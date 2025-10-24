"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { TextEmbeddingInputOutput } from "@/components/text-embedding-input-output";

interface EmbeddingResult {
  request_id: string;
  embeddings: number[][];
  dimensions: number;
  model_used: string;
  processing_time_ms: number;
}

interface EmbeddingModel {
  id: string;
  label: string;
  provider: string;
  status: string;
  dimensions: number;
}

export default function EmbeddingPage() {
  const searchParams = useSearchParams();
  const [texts, setTexts] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EmbeddingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<EmbeddingModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);

  useEffect(() => {
    // Load available models
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/skills?task=feature-extraction");
        if (!response.ok) throw new Error("Failed to fetch skills");
        const data = await response.json();
        // Filter for ready skills only (task filter already applied)
        const downloadedModels = data.skills.filter(
          (m: EmbeddingModel) => m.status === "ready"
        );
        setAvailableModels(downloadedModels);

        // Check if skill query param is provided
        const skillParam = searchParams.get("skill");
        if (skillParam && downloadedModels.some((m: EmbeddingModel) => m.id === skillParam)) {
          // Pre-select the skill from query param if it exists and is ready
          setSelectedModel(skillParam);
        } else if (downloadedModels.length > 0 && !selectedModel) {
          // Select first downloaded model by default
          setSelectedModel(downloadedModels[0].id);
        }
      } catch (err) {
        console.error("Failed to fetch skills:", err);
      } finally {
        setModelsLoading(false);
      }
    };
    fetchModels();
  }, [searchParams, selectedModel]);

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
    </div>
  );
}
