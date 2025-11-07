"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { TextGenerationInputOutput } from "@/components/text-generation-input-output";

interface GenerationResult {
  request_id: string;
  generated_text: string;
  model: string;
  tokens_generated: number;
  processing_time_ms: number;
}

interface SkillInfo {
  id: string;
  label: string;
  provider: string;
  tasks: string[];
  architecture: string;
  default_prompt: string | null;
  size_mb: number;
  parameters_m: number;
  gpu_memory_mb: number;
  reference_url: string;
  status: string;
  downloaded_size_mb?: number;
  error_message?: string;
}

function TextGenerationContent() {
  const searchParams = useSearchParams();
  const [prompt, setPrompt] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<SkillInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Generation parameters
  const [maxNewTokens, setMaxNewTokens] = useState<number>(256);
  const [temperature, setTemperature] = useState<number>(0.7);
  const [topP, setTopP] = useState<number>(0.9);

  useEffect(() => {
    // Fetch available models
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/models?task=text-generation");
        if (!response.ok) {
          throw new Error("Failed to fetch models");
        }
        const data = await response.json();
        // Filter for ready models only in Try tab
        const downloadedModels = data.models.filter(
          (s: SkillInfo) => s.status === "ready"
        );
        setAvailableModels(downloadedModels);

        // Check if model or legacy skill query param is provided
        const skillParam = searchParams.get("skill") || searchParams.get("model");
        if (skillParam && downloadedModels.some((s: SkillInfo) => s.id === skillParam)) {
          // Pre-select the skill from query param if it exists and is ready
          setSelectedModel(skillParam);
        } else if (downloadedModels.length > 0) {
          // Set first downloaded model as default
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
  }, [searchParams]);

  const handleGenerate = async () => {
    if (!prompt.trim() || !selectedModel) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/text-generation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          model: selectedModel,
          max_new_tokens: maxNewTokens,
          temperature: temperature,
          top_p: topP,
          do_sample: true,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      toast.success("Text generated successfully!", {
        description: `${data.tokens_generated} tokens in ${data.processing_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to generate text";
      setError(errorMsg);
      toast.error("Text generation failed", {
        description: errorMsg,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Text Generation</h1>
        <p className="text-muted-foreground">
          Generate text completions using AI language models
        </p>
      </div>

      <TextGenerationInputOutput
        mode="try"
        prompt={prompt}
        onPromptChange={setPrompt}
        result={result}
        loading={loading}
        error={error}
        onSend={handleGenerate}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        availableModels={availableModels}
        modelsLoading={modelsLoading}
        maxNewTokens={maxNewTokens}
        onMaxNewTokensChange={setMaxNewTokens}
        temperature={temperature}
        onTemperatureChange={setTemperature}
        topP={topP}
        onTopPChange={setTopP}
      />
    </div>
  );
}

export default function TextGenerationPage() {
  return (
    <Suspense fallback={<div className="container mx-auto px-4 py-8"><div>Loading...</div></div>}>
      <TextGenerationContent />
    </Suspense>
  );
}
