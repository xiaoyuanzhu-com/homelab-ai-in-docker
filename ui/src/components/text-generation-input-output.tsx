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
import { Slider } from "@/components/ui/slider";
import { AlertCircle, ChevronDown, ChevronUp } from "lucide-react";

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
}: TextGenerationInputOutputProps) {
  const isEditable = mode === "try";
  const [isInputOpen, setIsInputOpen] = useState(false);
  const [isOutputOpen, setIsOutputOpen] = useState(false);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left: Input */}
      <Card>
        <CardHeader>
          <CardTitle>Input</CardTitle>
          <CardDescription>
            {isEditable ? "Enter a prompt to generate text" : "Input prompt"}
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

              {/* Advanced Parameters */}
              <div className="space-y-4 pt-2 border-t">
                <h4 className="text-sm font-medium">Advanced Parameters</h4>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="max-tokens" className="text-sm">Max New Tokens</Label>
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
                    <Label htmlFor="temperature" className="text-sm">Temperature</Label>
                    <span className="text-sm text-muted-foreground">{temperature.toFixed(2)}</span>
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
                    <Label htmlFor="top-p" className="text-sm">Top P</Label>
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
                  <p className="text-xs text-muted-foreground">
                    Nucleus sampling threshold
                  </p>
                </div>
              </div>
            </>
          )}

          {!isEditable && prompt && (
            <div className="space-y-2">
              <Label>Prompt</Label>
              <p className="text-sm p-4 bg-muted rounded-lg font-mono whitespace-pre-wrap">
                {prompt}
              </p>
            </div>
          )}

          {isEditable && onSend && (
            <Button
              onClick={onSend}
              disabled={loading || !prompt.trim() || !selectedModel}
              className="w-full"
            >
              {loading ? "Generating..." : "Generate Text"}
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
                    prompt: prompt || null,
                    max_new_tokens: maxNewTokens,
                    temperature: temperature,
                    top_p: topP,
                    do_sample: true,
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
          <CardDescription>Generated text</CardDescription>
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
