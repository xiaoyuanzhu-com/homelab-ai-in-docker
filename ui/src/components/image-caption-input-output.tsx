"use client";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { AlertCircle } from "lucide-react";
import { TryLayout } from "@/components/try/try-layout";
import { ModelSelector } from "@/components/try";
import { ImageUpload } from "@/components/inputs";

interface CaptionResult {
  request_id: string;
  caption: string;
  model: string;
  processing_time_ms: number;
}

interface ModelInfo {
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

interface ImageCaptionInputOutputProps {
  mode: "try" | "history";

  // Input state
  imagePreview: string;
  onImageChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;

  // Output/Result
  result: CaptionResult | null;
  loading: boolean;
  error: string | null;
  responsePayload?: Record<string, unknown>;

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

  // Overrides for history playback
  requestPayload?: Record<string, unknown>;
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
  requestPayload,
  responsePayload,
}: ImageCaptionInputOutputProps) {
  const isEditable = mode === "try";

  // Get the selected model's default prompt
  const selectedModelInfo = availableModels.find((m) => m.id === selectedModel);
  const defaultPrompt = selectedModelInfo?.default_prompt;

  const modelOptions = availableModels.map((model) => ({
    value: model.id,
    label: `${model.label} (${model.provider})`,
  }));

  const requestModel =
    typeof requestPayload?.model === "string" ? (requestPayload.model as string) : undefined;
  const resolvedModel = requestModel ?? result?.model ?? selectedModel ?? "";

  const rawRequestData =
    requestPayload ??
    {
      model: selectedModel || null,
      ...(defaultPrompt && { prompt: prompt || null }),
      image: imagePreview
        ? `<base64 data, ${Math.round(imagePreview.length / 1024)}KB>`
        : null,
    };

  const rawResponseData = responsePayload ?? result;

  const inputContent = isEditable ? (
    <>
      <div className="space-y-2">
        <Label htmlFor="model">Model</Label>
        <ModelSelector
          value={selectedModel ?? ""}
          onChange={onModelChange}
          options={modelOptions}
          loading={modelsLoading}
          disabled={loading}
          emptyMessage="No captioning models downloaded. Visit the Models page to install one."
        />
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
            Modify the prompt to ask different questions about the image.
          </p>
        </div>
      )}

      <ImageUpload
        id="image"
        label="Image"
        onChange={onImageChange}
        previewSrc={imagePreview}
        disabled={loading}
        helperText="Supported formats: JPG, PNG, WebP."
      />
    </>
  ) : (
    <>
      {resolvedModel && (
        <div className="space-y-2">
          <Label>Model</Label>
          <p className="text-sm font-mono bg-muted p-2 rounded">
            {resolvedModel}
          </p>
        </div>
      )}
      {prompt && (
        <div className="space-y-2">
          <Label>Prompt</Label>
          <p className="text-sm p-3 bg-muted rounded-lg whitespace-pre-wrap">
            {prompt}
          </p>
        </div>
      )}
      {imagePreview && (
        <ImageUpload label="Image" previewSrc={imagePreview} hideInput />
      )}
    </>
  );

  const outputContent = (
    <>
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

      {result && !error && (
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
    </>
  );

  return (
    <TryLayout
      input={{
        title: "Input",
        description: isEditable ? "Upload an image to caption" : "Input image",
        children: inputContent,
        footer:
          isEditable && onSend ? (
            <Button
              onClick={onSend}
              disabled={loading || !imagePreview || !selectedModel}
              className="w-full"
            >
              {loading ? "Generating..." : "Generate Caption"}
            </Button>
          ) : null,
        rawPayload: {
          label: "View Raw Request",
          payload: rawRequestData,
        },
      }}
      output={{
        title: "Output",
        description: "Generated caption",
        children: outputContent,
        rawPayload: {
          label: "View Raw Response",
          payload: rawResponseData,
        },
      }}
    />
  );
}
