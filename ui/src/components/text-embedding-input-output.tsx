"use client";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle } from "lucide-react";
import { TryLayout } from "@/components/try/try-layout";
import { ModelSelector } from "@/components/try";

interface EmbeddingModel {
  id: string;
  label: string;
  provider: string;
  status: string;
  dimensions: number;
}

interface EmbeddingResult {
  request_id: string;
  embeddings: number[][];
  dimensions: number;
  model_used: string;
  processing_time_ms: number;
}

interface TextEmbeddingInputOutputProps {
  mode: "try" | "history";

  // Input state
  texts: string;
  onTextsChange?: (texts: string) => void;

  // Model selection
  selectedModel: string;
  onModelChange?: (modelId: string) => void;
  availableModels: EmbeddingModel[];
  modelsLoading?: boolean;

  // Output/Result
  result: EmbeddingResult | null;
  loading: boolean;
  error: string | null;
  responsePayload?: Record<string, unknown>;

  // Actions
  onSend?: () => void;

  // Overrides for history playback
  requestPayload?: Record<string, unknown>;
}

export function TextEmbeddingInputOutput({
  mode,
  texts,
  onTextsChange,
  selectedModel,
  onModelChange,
  availableModels,
  modelsLoading = false,
  result,
  loading,
  error,
  onSend,
  requestPayload,
  responsePayload,
}: TextEmbeddingInputOutputProps) {
  const isEditable = mode === "try";
  const textList = texts.split("\n").filter((t) => t.trim());

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && e.ctrlKey && isEditable && onSend) {
      onSend();
    }
  };

  const modelOptions = availableModels.map((model) => ({
    value: model.id,
    label: `${model.label} (${model.provider}) - ${model.dimensions}d`,
  }));

  const rawRequestData = requestPayload ?? { texts: textList, model: selectedModel };
  const rawResponseData = responsePayload ?? result;

  const inputContent = (
    <>
      <div className="space-y-2">
        <Label htmlFor="model">Model</Label>
        <ModelSelector
          value={selectedModel}
          onChange={onModelChange}
          options={modelOptions}
          loading={modelsLoading}
          disabled={!isEditable}
          emptyMessage="No embedding models downloaded. Install one to continue."
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="texts">Texts</Label>
        <Textarea
          id="texts"
          placeholder={
            isEditable
              ? "Enter text here\nOne text per line\nExample: The quick brown fox"
              : ""
          }
          value={texts}
          onChange={(e) => onTextsChange?.(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={10}
          readOnly={!isEditable}
          className={!isEditable ? "bg-muted" : ""}
        />
        <p className="text-xs text-muted-foreground">{textList.length} text(s)</p>
      </div>
    </>
  );

  const outputContent = (
    <>
      {loading && (
        <div className="space-y-4">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-32 w-full" />
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

      {result && !error && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge>{result.embeddings.length} vectors</Badge>
            <Badge variant="outline">{result.dimensions} dimensions</Badge>
            <span className="text-sm text-muted-foreground">
              {result.processing_time_ms}ms
            </span>
          </div>

          <div>
            <Label>Model</Label>
            <p className="text-sm mt-1 font-mono">{result.model_used}</p>
          </div>

          {result.embeddings.length > 0 && (
            <div>
              <Label>Embeddings</Label>
              <div className="mt-2 p-4 bg-muted rounded-lg max-h-96 overflow-y-auto">
                {result.embeddings.map((embedding, idx) => (
                  <div key={idx} className="mb-4 last:mb-0">
                    <div className="text-xs text-muted-foreground mb-1">
                      Vector {idx + 1}:
                    </div>
                    <div className="text-xs font-mono break-all">
                      [{embedding.slice(0, 5).map((n) => n.toFixed(4)).join(", ")}
                      {embedding.length > 5 && `, ... (${embedding.length} total)`}]
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

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
            ? 'Enter texts and click "Generate Embeddings" to see results'
            : "No output available"}
        </p>
      )}
    </>
  );

  return (
    <TryLayout
      input={{
        title: "Input",
        description: isEditable ? "Enter texts to embed (one per line)" : "Input texts",
        children: inputContent,
        footer:
          isEditable && onSend ? (
            <Button
              onClick={onSend}
              disabled={loading || !texts.trim() || !selectedModel}
              className="w-full"
            >
              {loading ? "Generating..." : "Generate Embeddings"}
            </Button>
          ) : null,
        rawPayload: {
          label: "View Raw Request",
          payload: rawRequestData,
        },
      }}
      output={{
        title: "Output",
        description: "Embedding vectors",
        children: outputContent,
        rawPayload: {
          label: "View Raw Response",
          payload: rawResponseData,
        },
      }}
    />
  );
}
