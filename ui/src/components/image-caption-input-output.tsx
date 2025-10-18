"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { AlertCircle, ChevronDown, ChevronUp } from "lucide-react";

interface CaptionResult {
  request_id: string;
  caption: string;
  model_used: string;
  processing_time_ms: number;
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
}

export function ImageCaptionInputOutput({
  mode,
  imagePreview,
  onImageChange,
  result,
  loading,
  error,
  onSend,
}: ImageCaptionInputOutputProps) {
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
            {isEditable ? "Upload an image to caption" : "Input image"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {isEditable && (
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
              disabled={loading || !imagePreview}
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
