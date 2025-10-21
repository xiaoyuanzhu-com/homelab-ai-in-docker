"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { AlertCircle, ChevronDown, ChevronUp } from "lucide-react";

interface CaptionResult {
  request_id: string;
  caption: string;
  model: string;
  processing_time_ms: number;
}

interface ModelInfo {
  id: string;
  name: string;
  team: string;
  type: string;
  task: string;
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

interface ImageCaptionInputOutputProps {
  mode: "try" | "history";

  // Input state
  imagePreview: string;
  onImageChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;

  // Output/Result
  result: CaptionResult | null;
  loading: boolean;
  error: string | null;

  // Actions
  onSend?: () => void;

  // Model selection
  selectedModel?: string;
  onModelChange?: (model: string) => void;
  availableModels?: ModelInfo[];
  modelsLoading?: boolean;

  // Prompt
  prompt?: string;
  onPromptChange?: (prompt: string) => void;
}

export function ImageCaptionInputOutput({
  mode,
  imagePreview,
  onImageChange,
  result,
  loading,
  error,
  onSend,
  selectedModel,
  onModelChange,
  availableModels = [],
  modelsLoading = false,
  prompt = "",
  onPromptChange,
}: ImageCaptionInputOutputProps) {
  const isEditable = mode === "try";
  const [isInputOpen, setIsInputOpen] = useState(false);
  const [isOutputOpen, setIsOutputOpen] = useState(false);

  // Get the selected model's default prompt
  const selectedModelInfo = availableModels.find((m) => m.id === selectedModel);
  const defaultPrompt = selectedModelInfo?.default_prompt;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left: Input */}
      <Card>
        <CardHeader>
          <CardTitle>Input</CardTitle>
          <CardDescription>
            {isEditable ? "Upload an image to caption" : "Input image"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {isEditable && (
            <>
              <div className="space-y-2">
                <Label htmlFor="model">Model</Label>
                {modelsLoading ? (
                  <Skeleton className="h-10 w-full" />
                ) : availableModels.length === 0 ? (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      No models downloaded. Please download a model from the Models page first.
                    </AlertDescription>
                  </Alert>
                ) : (
                  <Select value={selectedModel} onValueChange={onModelChange} disabled={loading}>
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
              {defaultPrompt && (
                <div className="space-y-2">
                  <Label htmlFor="prompt">Prompt</Label>
                  <Textarea
                    id="prompt"
                    placeholder="Enter a custom prompt or question..."
                    value={prompt}
                    onChange={(e) => onPromptChange?.(e.target.value)}
                    disabled={loading}
                    rows={3}
                    className="resize-none"
                  />
                  <p className="text-xs text-muted-foreground">
                    You can modify the prompt to ask different questions about the image.
                  </p>
                </div>
              )}
              <div className="space-y-2">
                <Label htmlFor="image">Image</Label>
                <input
                  id="image"
                  type="file"
                  accept="image/*"
                  onChange={onImageChange}
                  className="block w-full text-sm text-muted-foreground file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                />
              </div>
            </>
          )}

          {imagePreview && (
            <div className="space-y-2">
              <Label>Preview</Label>
              <div className="border rounded-lg p-2 bg-muted/50">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="max-w-full h-auto max-h-96 mx-auto rounded"
                />
              </div>
            </div>
          )}

          {isEditable && onSend && (
            <Button
              onClick={onSend}
              disabled={loading || !imagePreview || !selectedModel}
              className="w-full"
            >
              {loading ? "Generating..." : "Generate Caption"}
            </Button>
          )}

          <Collapsible open={isInputOpen} onOpenChange={setIsInputOpen}>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="w-full justify-between">
                <span className="text-sm font-medium">View Raw Request</span>
                {isInputOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2">
              <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto max-h-48">
                {JSON.stringify(
                  {
                    model: selectedModel || null,
                    ...(defaultPrompt && { prompt: prompt || null }),
                    image: imagePreview ? `<base64 data, ${Math.round(imagePreview.length / 1024)}KB>` : null,
                  },
                  null,
                  2
                )}
              </pre>
            </CollapsibleContent>
          </Collapsible>
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
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-32 w-full" />
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
              <div className="flex items-center gap-2 flex-wrap">
                <Badge>Caption</Badge>
                <span className="text-sm text-muted-foreground">
                  {result.processing_time_ms}ms
                </span>
              </div>

              <div>
                <Label>Generated Caption</Label>
                <p className="text-sm mt-2 p-4 bg-muted rounded-lg">
                  {result.caption}
                </p>
              </div>

              <div>
                <Label>Model</Label>
                <p className="text-sm mt-1 font-mono">{result.model}</p>
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
              {isEditable
                ? 'Upload an image and click "Generate Caption" to see results'
                : "No output available"}
            </p>
          )}

          {result && (
            <Collapsible open={isOutputOpen} onOpenChange={setIsOutputOpen} className="mt-4">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="w-full justify-between">
                  <span className="text-sm font-medium">View Raw Response</span>
                  {isOutputOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="pt-2">
                <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto max-h-96">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </CollapsibleContent>
            </Collapsible>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
