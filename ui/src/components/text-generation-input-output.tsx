"use client";

import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { AlertCircle } from "lucide-react";
import { TryLayout } from "@/components/try/try-layout";
import { ModelSelector } from "@/components/try";
import type { RawPayloadProps } from "@/components/try/raw-payload";

interface GenerationResult {
  request_id: string;
  generated_text: string;
  model: string;
  tokens_generated: number;
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

interface TextGenerationInputOutputProps {
  mode: "try" | "history";

  // Input state
  prompt: string;
  onPromptChange?: (prompt: string) => void;

  // Output/Result
  result: GenerationResult | null;
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

  // Generation parameters
  maxNewTokens?: number;
  onMaxNewTokensChange?: (value: number) => void;
  temperature?: number;
  onTemperatureChange?: (value: number) => void;
  topP?: number;
  onTopPChange?: (value: number) => void;

  // Overrides for history playback
  requestPayload?: Record<string, unknown>;
}

export function TextGenerationInputOutput({
  mode,
  prompt,
  onPromptChange,
  result,
  loading,
  error,
  onSend,
  selectedModel,
  onModelChange,
  availableModels = [],
  modelsLoading = false,
  maxNewTokens = 256,
  onMaxNewTokensChange,
  temperature = 0.7,
  onTemperatureChange,
  topP = 0.9,
  onTopPChange,
  requestPayload,
  responsePayload,
}: TextGenerationInputOutputProps) {
  const isEditable = mode === "try";

  const modelOptions = availableModels.map((model) => ({
    value: model.id,
    label: `${model.name} (${model.team})`,
  }));

  const defaultRequestPayload = {
    model: selectedModel || null,
    prompt: prompt || null,
    max_new_tokens: maxNewTokens,
    temperature: temperature,
    top_p: topP,
    do_sample: true,
  };

  const mergedRequestPayload: RawPayloadProps["payload"] =
    requestPayload ?? defaultRequestPayload;

  const mergedResponsePayload: RawPayloadProps["payload"] =
    responsePayload ?? result ?? null;

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
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="prompt">Prompt</Label>
        <Textarea
          id="prompt"
          placeholder="Enter your prompt here..."
          value={prompt}
          onChange={(e) => onPromptChange?.(e.target.value)}
          disabled={loading}
          rows={8}
          className="resize-none font-mono text-sm"
        />
      </div>

      <div className="space-y-4 pt-2 border-t">
        <h4 className="text-sm font-medium">Advanced Parameters</h4>

        <div className="space-y-2">
          <div className="flex justify-between">
            <Label htmlFor="max-tokens" className="text-sm">
              Max New Tokens
            </Label>
            <span className="text-sm text-muted-foreground">{maxNewTokens}</span>
          </div>
          <Slider
            id="max-tokens"
            min={1}
            max={2048}
            step={1}
            value={[maxNewTokens]}
            onValueChange={(value) => onMaxNewTokensChange?.(value[0])}
            disabled={loading}
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <Label htmlFor="temperature" className="text-sm">
              Temperature
            </Label>
            <span className="text-sm text-muted-foreground">
              {temperature.toFixed(2)}
            </span>
          </div>
          <Slider
            id="temperature"
            min={0}
            max={2}
            step={0.01}
            value={[temperature]}
            onValueChange={(value) => onTemperatureChange?.(value[0])}
            disabled={loading}
          />
          <p className="text-xs text-muted-foreground">
            Higher values make output more random, lower values more deterministic
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <Label htmlFor="top-p" className="text-sm">
              Top P
            </Label>
            <span className="text-sm text-muted-foreground">{topP.toFixed(2)}</span>
          </div>
          <Slider
            id="top-p"
            min={0}
            max={1}
            step={0.01}
            value={[topP]}
            onValueChange={(value) => onTopPChange?.(value[0])}
            disabled={loading}
          />
          <p className="text-xs text-muted-foreground">Nucleus sampling threshold</p>
        </div>
      </div>
    </>
  ) : (
    <>
      {prompt && (
        <div className="space-y-2">
          <Label>Prompt</Label>
          <p className="text-sm p-4 bg-muted rounded-lg font-mono whitespace-pre-wrap">
            {prompt}
          </p>
        </div>
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
            <Badge>Generated</Badge>
            <span className="text-sm text-muted-foreground">
              {result.tokens_generated} tokens in {result.processing_time_ms}ms
            </span>
          </div>

          <div>
            <Label>Generated Text</Label>
            <p className="text-sm mt-2 p-4 bg-muted rounded-lg whitespace-pre-wrap font-mono">
              {result.generated_text}
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
            ? 'Enter a prompt and click "Generate Text" to see results'
            : "No output available"}
        </p>
      )}
    </>
  );

  return (
    <TryLayout
      input={{
        title: "Input",
        description: isEditable ? "Enter a prompt to generate text" : "Input prompt",
        children: inputContent,
        footer:
          isEditable && onSend ? (
            <Button
              onClick={onSend}
              disabled={loading || !prompt.trim() || !selectedModel}
              className="w-full"
            >
              {loading ? "Generating..." : "Generate Text"}
            </Button>
          ) : null,
        rawPayload: {
          label: "View Raw Request",
          payload: mergedRequestPayload,
        },
      }}
      output={{
        title: "Output",
        description: "Generated text",
        children: outputContent,
        rawPayload: {
          label: "View Raw Response",
          payload: mergedResponsePayload,
        },
      }}
    />
  );
}
