"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { ImageCaptionInputOutput } from "@/components/image-caption-input-output";

interface CaptionResult {
  request_id: string;
  caption: string;
  model: string;
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

export default function ImageCaptionPage() {
  const searchParams = useSearchParams();
  const [, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CaptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<SkillInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [prompt, setPrompt] = useState<string>("");

  useEffect(() => {
    // Fetch available skills
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/skills?task=image-captioning");
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
        let modelToSelect: SkillInfo | null = null;

        if (skillParam && downloadedModels.some((s: SkillInfo) => s.id === skillParam)) {
          // Pre-select the skill from query param if it exists and is ready
          modelToSelect = downloadedModels.find((s: SkillInfo) => s.id === skillParam) || null;
          if (modelToSelect) {
            setSelectedModel(modelToSelect.id);
          }
        } else if (downloadedModels.length > 0) {
          // Set first downloaded model as default
          modelToSelect = downloadedModels[0];
          setSelectedModel(modelToSelect.id);
        }

        // Set default prompt if model requires it
        if (modelToSelect?.default_prompt) {
          setPrompt(modelToSelect.default_prompt);
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
    // Update prompt when model changes
    if (selectedModel && availableModels.length > 0) {
      const model = availableModels.find((m) => m.id === selectedModel);
      if (model?.default_prompt) {
        setPrompt(model.default_prompt);
      } else {
        setPrompt("");
      }
    }
  }, [selectedModel, availableModels]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleCaption = async () => {
    if (!imagePreview || !selectedModel) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Extract base64 from data URL
      const base64Data = imagePreview.split(",")[1];

      const response = await fetch("/api/image-captioning", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: base64Data,
          model: selectedModel,
          prompt: prompt || undefined,  // Only include if non-empty
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      toast.success("Caption generated successfully!", {
        description: `Processed in ${data.processing_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to generate caption";
      setError(errorMsg);
      toast.error("Caption generation failed", {
        description: errorMsg,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Image Captioning</h1>
        <p className="text-muted-foreground">
          Generate descriptive captions for images using AI
        </p>
      </div>

      <ImageCaptionInputOutput
        mode="try"
        imagePreview={imagePreview}
        onImageChange={handleFileChange}
        result={result}
        loading={loading}
        error={error}
        onSend={handleCaption}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        availableModels={availableModels}
        modelsLoading={modelsLoading}
        prompt={prompt}
        onPromptChange={setPrompt}
      />
    </div>
  );
}
