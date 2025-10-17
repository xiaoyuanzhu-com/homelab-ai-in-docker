"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { HistoryPanel } from "@/components/history-panel";

interface CaptionResult {
  request_id: string;
  caption: string;
  model_used: string;
  processing_time_ms: number;
}

export default function ImageCaptionPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [activeTab, setActiveTab] = useState(searchParams.get("tab") || "try");

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CaptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    router.push(`/image-caption?tab=${tab}`, { scroll: false });
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
    if (!imagePreview) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Extract base64 from data URL
      const base64Data = imagePreview.split(",")[1];

      const response = await fetch("/api/caption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Data }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
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
        <h1 className="text-3xl font-bold mb-2">Image Caption</h1>
        <p className="text-muted-foreground">
          Generate descriptive captions for images using AI
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="mb-6">
          <TabsTrigger value="try">Try</TabsTrigger>
          <TabsTrigger value="api">API</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Try Tab */}
        <TabsContent value="try">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Input */}
            <Card>
              <CardHeader>
                <CardTitle>Input</CardTitle>
                <CardDescription>Upload an image to caption</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="image">Image</Label>
                  <input
                    id="image"
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-muted-foreground
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-md file:border-0
                      file:text-sm file:font-semibold
                      file:bg-primary file:text-primary-foreground
                      hover:file:bg-primary/90"
                  />
                </div>

                {imagePreview && (
                  <div className="space-y-2">
                    <Label>Preview</Label>
                    <div className="border rounded-lg p-2">
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="max-w-full h-auto max-h-64 mx-auto"
                      />
                    </div>
                  </div>
                )}

                <Button
                  onClick={handleCaption}
                  disabled={loading || !imagePreview}
                  className="w-full"
                >
                  {loading ? "Generating..." : "Generate Caption"}
                </Button>
              </CardContent>
            </Card>

            {/* Right: Output */}
            <Card>
              <CardHeader>
                <CardTitle>Output</CardTitle>
                <CardDescription>Generated caption</CardDescription>
              </CardHeader>
              <CardContent>
                {loading && (
                  <div className="space-y-4">
                    <Skeleton className="h-8 w-full" />
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                )}

                {error && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {result && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Badge>Success</Badge>
                      <span className="text-sm text-muted-foreground">
                        {result.processing_time_ms}ms
                      </span>
                    </div>

                    <div>
                      <Label>Caption</Label>
                      <div className="mt-2 p-4 bg-muted rounded-lg">
                        <p className="text-lg">{result.caption}</p>
                      </div>
                    </div>

                    <div>
                      <Label>Model</Label>
                      <p className="text-sm mt-1 font-mono">{result.model_used}</p>
                    </div>

                    <div>
                      <Label>Request ID</Label>
                      <p className="text-xs text-muted-foreground font-mono mt-1">
                        {result.request_id}
                      </p>
                    </div>
                  </div>
                )}

                {!loading && !error && !result && (
                  <p className="text-muted-foreground text-center py-8">
                    Upload an image and click "Generate Caption" to see results
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* API Tab */}
        <TabsContent value="api">
          <Card>
            <CardHeader>
              <CardTitle>API Reference</CardTitle>
              <CardDescription>HTTP endpoint details and examples</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">Endpoint</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <code className="text-sm">POST /api/caption</code>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Request Body</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "image": "base64_encoded_image_data",
  "model": "Salesforce/blip-image-captioning-base"
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Parameters</h3>
                <ul className="space-y-2 text-sm">
                  <li><code className="bg-muted px-2 py-1 rounded">image</code> (string, required) - Base64-encoded image data</li>
                  <li><code className="bg-muted px-2 py-1 rounded">model</code> (string, optional) - Model name (default: Salesforce/blip-image-captioning-base)</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Response</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "request_id": "uuid",
  "caption": "a dog sitting on a bench",
  "model_used": "Salesforce/blip-image-captioning-base",
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
    "http://localhost:8000/api/caption",
    json={"image": image_data}
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
          <HistoryPanel
            service="caption"
            onSelectEntry={() => {
              setActiveTab("try");
              router.push("/image-caption?tab=try", { scroll: false });
            }}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
