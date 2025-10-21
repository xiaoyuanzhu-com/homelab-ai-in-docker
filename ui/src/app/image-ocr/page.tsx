"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { FileText, Loader2, Upload, Copy } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface OCRResult {
  request_id: string;
  text: string;
  model: string;
  processing_time_ms: number;
  output_format: string;
}

interface ModelInfo {
  id: string;
  name: string;
  team: string;
  type: string;
  task: string;
  architecture: string;
  size_mb: number;
  parameters_m: number;
  gpu_memory_mb: number;
  link: string;
  status: string;
  downloaded_size_mb?: number;
  error_message?: string;
}

export default function ImageOCRPage() {
  const [activeTab, setActiveTab] = useState("try");

  const [, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OCRResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [outputFormat, setOutputFormat] = useState<"text" | "markdown">("text");
  const [viewMode, setViewMode] = useState<"raw" | "rendered">("rendered");

  useEffect(() => {
    // Infer API base URL from current window location
    if (typeof window !== "undefined") {
      setApiBaseUrl(window.location.origin);
    }
  }, []);

  useEffect(() => {
    // Fetch available models
    const fetchModels = async () => {
      try {
        const response = await fetch("/api/models?task=image-ocr");
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

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
  };

  // Check if selected model supports markdown
  const supportsMarkdown = () => {
    if (!selectedModel) return false;
    const model = availableModels.find((m) => m.id === selectedModel);
    if (!model) return false;
    // Granite Docling, MinerU, and DeepSeek support markdown
    return ["granite-docling", "mineru", "deepseek"].includes(model.architecture);
  };

  const copyToClipboard = async () => {
    if (!result?.text) return;
    try {
      await navigator.clipboard.writeText(result.text);
      toast.success("Copied to clipboard!");
    } catch (err) {
      toast.error("Failed to copy to clipboard");
    }
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

  const handleOCR = async () => {
    if (!imagePreview || !selectedModel) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Extract base64 from data URL
      const base64Data = imagePreview.split(",")[1];

      const response = await fetch("/api/image-ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: base64Data,
          model: selectedModel,
          output_format: outputFormat,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail?.message || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      toast.success("Text extracted successfully!", {
        description: `Processed in ${data.processing_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to extract text";
      setError(errorMsg);
      toast.error("OCR failed", {
        description: errorMsg,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Image OCR</h1>
        <p className="text-muted-foreground">
          Extract text from images using optical character recognition
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="mb-6">
          <TabsTrigger value="try">Try</TabsTrigger>
          <TabsTrigger value="doc">Doc</TabsTrigger>
        </TabsList>

        {/* Try Tab */}
        <TabsContent value="try">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Input Card */}
            <Card>
              <CardHeader>
                <CardTitle>Input</CardTitle>
                <CardDescription>Upload an image to extract text</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Model Selection */}
                <div className="space-y-2">
                  <Label htmlFor="model">Model</Label>
                  {modelsLoading ? (
                    <div className="text-sm text-muted-foreground">Loading models...</div>
                  ) : availableModels.length === 0 ? (
                    <div className="text-sm text-muted-foreground">
                      No OCR models downloaded. Visit the{" "}
                      <a href="/models" className="text-primary hover:underline">
                        Models page
                      </a>{" "}
                      to download one.
                    </div>
                  ) : (
                    <Select value={selectedModel} onValueChange={setSelectedModel}>
                      <SelectTrigger id="model">
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {availableModels.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            {model.name} ({model.team})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </div>

                {/* Output Format */}
                <div className="space-y-2">
                  <Label htmlFor="format">Output Format</Label>
                  <Select
                    value={outputFormat}
                    onValueChange={(value) => setOutputFormat(value as "text" | "markdown")}
                    disabled={!supportsMarkdown()}
                  >
                    <SelectTrigger id="format">
                      <SelectValue placeholder="Select format" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="text">Plain Text</SelectItem>
                      <SelectItem value="markdown">Markdown</SelectItem>
                    </SelectContent>
                  </Select>
                  {!supportsMarkdown() && selectedModel && (
                    <p className="text-xs text-muted-foreground">
                      ⚠️ This model only supports plain text output
                    </p>
                  )}
                  {supportsMarkdown() && (
                    <p className="text-xs text-muted-foreground">
                      ✓ Markdown format supported by this model
                    </p>
                  )}
                </div>

                {/* Image Upload */}
                <div className="space-y-2">
                  <Label htmlFor="image">Image</Label>
                  <div className="border-2 border-dashed rounded-lg p-4 hover:border-primary/50 transition-colors">
                    <input
                      id="image"
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <label
                      htmlFor="image"
                      className="flex flex-col items-center justify-center cursor-pointer"
                    >
                      {imagePreview ? (
                        <img
                          src={imagePreview}
                          alt="Preview"
                          className="max-h-64 rounded-md"
                        />
                      ) : (
                        <>
                          <Upload className="h-8 w-8 text-muted-foreground mb-2" />
                          <span className="text-sm text-muted-foreground">
                            Click to upload image
                          </span>
                        </>
                      )}
                    </label>
                  </div>
                </div>

                {/* Extract Button */}
                <Button
                  onClick={handleOCR}
                  disabled={!imagePreview || !selectedModel || loading}
                  className="w-full"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Extracting...
                    </>
                  ) : (
                    <>
                      <FileText className="mr-2 h-4 w-4" />
                      Extract Text
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Output Card */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Output</CardTitle>
                    <CardDescription>Extracted text from the image</CardDescription>
                  </div>
                  {result && result.output_format === "markdown" && (
                    <div className="flex gap-2">
                      <Button
                        variant={viewMode === "rendered" ? "default" : "outline"}
                        size="sm"
                        onClick={() => setViewMode("rendered")}
                      >
                        Rendered
                      </Button>
                      <Button
                        variant={viewMode === "raw" ? "default" : "outline"}
                        size="sm"
                        onClick={() => setViewMode("raw")}
                      >
                        Raw
                      </Button>
                    </div>
                  )}
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {error && (
                  <div className="bg-destructive/10 text-destructive p-4 rounded-lg text-sm">
                    {error}
                  </div>
                )}

                {result && (
                  <>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Extracted Text</Label>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={copyToClipboard}
                          className="h-8"
                        >
                          <Copy className="h-4 w-4 mr-2" />
                          Copy
                        </Button>
                      </div>

                      {result.output_format === "markdown" && viewMode === "rendered" ? (
                        <div className="border rounded-lg p-4 min-h-[300px] prose prose-sm max-w-none dark:prose-invert">
                          <ReactMarkdown>{result.text}</ReactMarkdown>
                        </div>
                      ) : (
                        <Textarea
                          value={result.text}
                          readOnly
                          className="min-h-[300px] font-mono text-sm"
                          placeholder="Extracted text will appear here..."
                        />
                      )}
                    </div>

                    <div className="text-xs text-muted-foreground space-y-1">
                      <div>Request ID: {result.request_id}</div>
                      <div>Model: {result.model}</div>
                      <div>Output Format: {result.output_format}</div>
                      <div>Processing Time: {result.processing_time_ms}ms</div>
                    </div>
                  </>
                )}

                {!result && !error && (
                  <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                    <FileText className="h-12 w-12 mb-2" />
                    <p className="text-sm">Upload an image and click Extract Text</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
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
                  <code className="text-sm">POST /api/image-ocr</code>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Request Body</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "image": "base64_encoded_image_data",
  "model": "ibm-granite/granite-docling-258M",
  "output_format": "markdown"
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Parameters</h3>
                <ul className="space-y-2 text-sm">
                  <li><code className="bg-muted px-2 py-1 rounded">image</code> (string, required) - Base64-encoded image data</li>
                  <li><code className="bg-muted px-2 py-1 rounded">model</code> (string, required) - Model ID to use for OCR</li>
                  <li><code className="bg-muted px-2 py-1 rounded">output_format</code> (string, optional) - Output format: "text" (default) or "markdown" (supported by Granite Docling, MinerU, DeepSeek)</li>
                  <li><code className="bg-muted px-2 py-1 rounded">language</code> (string, optional) - Language hint for OCR (e.g., "en", "zh", "auto")</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Response</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "request_id": "uuid",
  "text": "# Document Title\\n\\nExtracted text from the image...",
  "model": "ibm-granite/granite-docling-258M",
  "output_format": "markdown",
  "processing_time_ms": 456
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Python Example</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`import base64
import requests

# Read and encode image
with open("document.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send request with markdown output
response = requests.post(
    "${apiBaseUrl}/api/image-ocr",
    json={
        "image": image_data,
        "model": "ibm-granite/granite-docling-258M",
        "output_format": "markdown"  # or "text" for plain text
    }
)

result = response.json()
print(result["text"])

# Save markdown to file
if result["output_format"] == "markdown":
    with open("output.md", "w") as f:
        f.write(result["text"])`}
                </pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
