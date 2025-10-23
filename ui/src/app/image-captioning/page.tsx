"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { ImageCaptionInputOutput } from "@/components/image-caption-input-output";
import { ImageCaptionHistory } from "@/components/image-caption-history";

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
  link: string;
  status: string;
  downloaded_size_mb?: number;
  error_message?: string;
}

export default function ImageCaptionPage() {
  const [activeTab, setActiveTab] = useState("try");

  const [, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CaptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<SkillInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [prompt, setPrompt] = useState<string>("");

  useEffect(() => {
    // Infer API base URL from current window location
    if (typeof window !== "undefined") {
      setApiBaseUrl(window.location.origin);
    }
  }, []);

  useEffect(() => {
    // Fetch available skills
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/skills?task=image-captioning");
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
          const firstModel = downloadedModels[0];
          setSelectedModel(firstModel.id);
          // Set default prompt if model requires it
          if (firstModel.default_prompt) {
            setPrompt(firstModel.default_prompt);
          }
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

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
  };

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

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="mb-6">
          <TabsTrigger value="try">Try</TabsTrigger>
          <TabsTrigger value="doc">Doc</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Try Tab */}
        <TabsContent value="try">
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
                  <code className="text-sm">POST /api/image-captioning</code>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Request Body</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "image": "base64_encoded_image_data",
  "model": "Salesforce/blip-image-captioning-base",
  "prompt": "Describe this image in detail"
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Parameters</h3>
                <ul className="space-y-2 text-sm">
                  <li><code className="bg-muted px-2 py-1 rounded">image</code> (string, required) - Base64-encoded image data</li>
                  <li><code className="bg-muted px-2 py-1 rounded">model</code> (string, required) - Model ID to use for captioning</li>
                  <li><code className="bg-muted px-2 py-1 rounded">prompt</code> (string, optional) - Custom prompt or question for the model. If not provided, uses model&apos;s default.</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Response</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "request_id": "uuid",
  "caption": "a dog sitting on a bench",
  "model": "Salesforce/blip-image-captioning-base",
  "processing_time_ms": 234
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Python Example</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`import base64
import requests

# Read and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "${apiBaseUrl}/api/image-captioning",
    json={
        "image": image_data,
        "model": "Salesforce/blip-image-captioning-base",
        "prompt": "What objects are in this image?"  # Optional
    }
)

result = response.json()
print(result["caption"])`}
                </pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history">
          <ImageCaptionHistory />
        </TabsContent>
      </Tabs>
    </div>
  );
}
