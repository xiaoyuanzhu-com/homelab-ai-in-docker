"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { TextGenerationInputOutput } from "@/components/text-generation-input-output";
import { TextGenerationHistory } from "@/components/text-generation-history";

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
  link: string;
  status: string;
  downloaded_size_mb?: number;
  error_message?: string;
}

export default function TextGenerationPage() {
  const [activeTab, setActiveTab] = useState("try");

  const [prompt, setPrompt] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<SkillInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Generation parameters
  const [maxNewTokens, setMaxNewTokens] = useState<number>(256);
  const [temperature, setTemperature] = useState<number>(0.7);
  const [topP, setTopP] = useState<number>(0.9);

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
        const response = await fetch("/api/skills?task=text-generation");
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

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
  };

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

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="mb-6">
          <TabsTrigger value="try">Try</TabsTrigger>
          <TabsTrigger value="doc">Doc</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Try Tab */}
        <TabsContent value="try">
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
                  <code className="text-sm">POST /api/text-generation</code>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Request Body</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "prompt": "Write a short story about a robot:",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Parameters</h3>
                <ul className="space-y-2 text-sm">
                  <li><code className="bg-muted px-2 py-1 rounded">prompt</code> (string, required) - Input prompt for text generation</li>
                  <li><code className="bg-muted px-2 py-1 rounded">model</code> (string, required) - Model ID to use for generation</li>
                  <li><code className="bg-muted px-2 py-1 rounded">max_new_tokens</code> (number, optional) - Maximum number of tokens to generate (default: 256, max: 4096)</li>
                  <li><code className="bg-muted px-2 py-1 rounded">temperature</code> (number, optional) - Sampling temperature, 0.0 = greedy, higher = more random (default: 0.7, range: 0.0-2.0)</li>
                  <li><code className="bg-muted px-2 py-1 rounded">top_p</code> (number, optional) - Nucleus sampling threshold (default: 0.9, range: 0.0-1.0)</li>
                  <li><code className="bg-muted px-2 py-1 rounded">do_sample</code> (boolean, optional) - Whether to use sampling vs greedy decoding (default: true)</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Response</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "request_id": "uuid",
  "generated_text": "Once upon a time, there was a robot...",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "tokens_generated": 128,
  "processing_time_ms": 1234
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Python Example</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`import requests

# Send request
response = requests.post(
    "${apiBaseUrl}/api/text-generation",
    json={
        "prompt": "Write a short story about a robot:",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
)

result = response.json()
print(result["generated_text"])`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">cURL Example</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`curl -X POST "${apiBaseUrl}/api/text-generation" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Write a short story about a robot:",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true
  }'`}
                </pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history">
          <TextGenerationHistory />
        </TabsContent>
      </Tabs>
    </div>
  );
}
