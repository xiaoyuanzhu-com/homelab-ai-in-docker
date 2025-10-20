"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { AlertCircle, ChevronDown, ChevronUp } from "lucide-react";

interface EmbeddingModel {
  id: string;
  name: string;
  team: string;
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

  // Actions
  onSend?: () => void;
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
}: TextEmbeddingInputOutputProps) {
  const isEditable = mode === "try";
  const textList = texts.split("\n").filter((t) => t.trim());
  const [isInputOpen, setIsInputOpen] = useState(false);
  const [isOutputOpen, setIsOutputOpen] = useState(false);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && e.ctrlKey && isEditable && onSend) {
      onSend();
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left: Input */}
      <Card>
        <CardHeader>
          <CardTitle>Input</CardTitle>
          <CardDescription>
            {isEditable ? "Enter texts to embed (one per line)" : "Input texts"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="model">Model</Label>
            {modelsLoading ? (
              <Skeleton className="h-10 w-full" />
            ) : availableModels.length === 0 ? (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  No models downloaded. Please go to the Models tab to download a model first.
                </AlertDescription>
              </Alert>
            ) : (
              <Select
                value={selectedModel}
                onValueChange={onModelChange}
                disabled={!isEditable}
              >
                <SelectTrigger id="model">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name} ({model.team}) - {model.dimensions}d
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="texts">Texts</Label>
            <Textarea
              id="texts"
              placeholder={isEditable ? "Enter text here&#10;One text per line&#10;Example: The quick brown fox" : ""}
              value={texts}
              onChange={(e) => onTextsChange?.(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={10}
              readOnly={!isEditable}
              className={!isEditable ? "bg-muted" : ""}
            />
            <p className="text-xs text-muted-foreground">
              {textList.length} text(s)
            </p>
          </div>

          {isEditable && onSend && (
            <Button
              onClick={onSend}
              disabled={loading || !texts.trim() || !selectedModel}
              className="w-full"
            >
              {loading ? "Generating..." : "Generate Embeddings"}
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
              <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto">
                {JSON.stringify(
                  {
                    texts: textList,
                    model: selectedModel,
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
          <CardDescription>Embedding vectors</CardDescription>
        </CardHeader>
        <CardContent>
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

          {result && (
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

              <div>
                <Label>Embeddings</Label>
                <div className="mt-2 p-4 bg-muted rounded-lg max-h-96 overflow-y-auto">
                  {result.embeddings.map((embedding, idx) => (
                    <div key={idx} className="mb-4 last:mb-0">
                      <div className="text-xs text-muted-foreground mb-1">
                        Vector {idx + 1}:
                      </div>
                      <div className="text-xs font-mono break-all">
                        [{embedding.slice(0, 5).map(n => n.toFixed(4)).join(", ")}
                        {embedding.length > 5 && `, ... (${embedding.length} total)`}]
                      </div>
                    </div>
                  ))}
                </div>
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
                ? 'Enter texts and click "Generate Embeddings" to see results'
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
