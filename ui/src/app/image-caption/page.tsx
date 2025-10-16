"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { HistoryPanel } from "@/components/history-panel";

interface CaptionResult {
  request_id: string;
  caption: string;
  model_used: string;
  processing_time_ms: number;
}

export default function ImageCaptionPage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CaptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate caption");
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                Generating caption...
              </div>
            )}

            {error && (
              <div className="text-destructive">
                <strong>Error:</strong> {error}
              </div>
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

        {/* History Panel */}
        <HistoryPanel service="caption" />
      </div>
    </div>
  );
}
