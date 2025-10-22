"use client";

import { ReactNode, useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronUp, RefreshCcw } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { TryLayout } from "@/components/try/try-layout";
import { TextGenerationInputOutput } from "@/components/text-generation-input-output";
import { ImageCaptionInputOutput } from "@/components/image-caption-input-output";
import { TextEmbeddingInputOutput } from "@/components/text-embedding-input-output";
import { CrawlInputOutput } from "@/components/crawl-input-output";

interface TaskHistoryEntry {
  service: string;
  timestamp: string;
  request_id: string;
  status: string;
  request: Record<string, unknown>;
  response: Record<string, unknown>;
}

type TaskRenderer = (entry: TaskHistoryEntry) => ReactNode;

const SERVICE_LABELS: Record<string, string> = {
  "text-generation": "Text Generation",
  "image-captioning": "Image Captioning",
  "text-to-embedding": "Text Embedding",
  crawl: "Web Crawling",
  "image-ocr": "Image OCR",
  "automatic-speech-recognition": "Speech Recognition",
};

const FILTER_OPTIONS = [
  { value: "all", label: "All tasks" },
  ...Object.entries(SERVICE_LABELS).map(([key, label]) => ({
    value: key,
    label,
  })),
];

function renderTextGeneration(entry: TaskHistoryEntry) {
  const prompt =
    typeof entry.request.prompt === "string" ? (entry.request.prompt as string) : "";

  const result =
    typeof entry.response.generated_text === "string"
      ? {
          request_id: entry.request_id,
          generated_text: entry.response.generated_text as string,
          model:
            typeof entry.response.model === "string" ? (entry.response.model as string) : "",
          tokens_generated:
            typeof entry.response.tokens_generated === "number"
              ? (entry.response.tokens_generated as number)
              : 0,
          processing_time_ms:
            typeof entry.response.processing_time_ms === "number"
              ? (entry.response.processing_time_ms as number)
              : 0,
        }
      : null;

  const error =
    entry.status === "success"
      ? null
      : typeof entry.response.error === "string"
      ? (entry.response.error as string)
      : "Request failed";

  return (
    <TextGenerationInputOutput
      mode="history"
      prompt={prompt}
      result={result}
      loading={false}
      error={error}
      requestPayload={entry.request}
      responsePayload={entry.response}
    />
  );
}

function renderImageCaption(entry: TaskHistoryEntry) {
  const imageData = typeof entry.request.image === "string" ? (entry.request.image as string) : "";
  const hasDataUri = imageData.startsWith("data:");
  const imagePreview = hasDataUri
    ? imageData
    : imageData
    ? `data:image/png;base64,${imageData}`
    : "";

  const prompt =
    typeof entry.request.prompt === "string" ? (entry.request.prompt as string) : "";

  const result =
    typeof entry.response.caption === "string"
      ? {
          request_id: entry.request_id,
          caption: entry.response.caption as string,
          model:
            typeof entry.response.model === "string" ? (entry.response.model as string) : "",
          processing_time_ms:
            typeof entry.response.processing_time_ms === "number"
              ? (entry.response.processing_time_ms as number)
              : 0,
        }
      : null;

  return (
    <ImageCaptionInputOutput
      mode="history"
      imagePreview={imagePreview}
      result={result}
      loading={false}
      error={entry.status === "success" ? null : "Request failed"}
      prompt={prompt}
      requestPayload={entry.request}
      responsePayload={entry.response}
    />
  );
}

function renderEmbeddings(entry: TaskHistoryEntry) {
  const texts = Array.isArray(entry.request.texts)
    ? (entry.request.texts as string[]).join("\n")
    : "";
  const selectedModel =
    typeof entry.request.model === "string"
      ? (entry.request.model as string)
      : typeof entry.response.model_used === "string"
      ? (entry.response.model_used as string)
      : "";

  const numEmbeddings =
    typeof entry.response.num_embeddings === "number"
      ? (entry.response.num_embeddings as number)
      : Array.isArray(entry.request.texts)
      ? (entry.request.texts as unknown[]).length
      : 0;

  const dimensions =
    typeof entry.response.dimensions === "number"
      ? (entry.response.dimensions as number)
      : 0;

  const embeddings = Array.from({ length: numEmbeddings }, () => [] as number[]);

  const result =
    typeof entry.response.dimensions === "number"
      ? {
          request_id: entry.request_id,
          embeddings,
          dimensions,
          model_used: selectedModel,
          processing_time_ms:
            typeof entry.response.processing_time_ms === "number"
              ? (entry.response.processing_time_ms as number)
              : 0,
        }
      : null;

  return (
    <TextEmbeddingInputOutput
      mode="history"
      texts={texts}
      selectedModel={selectedModel}
      availableModels={selectedModel ? [{ id: selectedModel, name: selectedModel, team: "", status: "downloaded", dimensions: result?.dimensions ?? 0 }] : []}
      result={result}
      loading={false}
      error={entry.status === "success" ? null : "Request failed"}
      requestPayload={entry.request}
      responsePayload={entry.response}
    />
  );
}

function renderCrawl(entry: TaskHistoryEntry) {
  const url = typeof entry.request.url === "string" ? (entry.request.url as string) : "";
  const waitForJs =
    typeof entry.request.wait_for_js === "boolean"
      ? (entry.request.wait_for_js as boolean)
      : true;
  const chromeCdpUrl =
    typeof entry.request.chrome_cdp_url === "string"
      ? (entry.request.chrome_cdp_url as string)
      : undefined;

  const result =
    typeof entry.response.url === "string"
      ? {
          request_id: entry.request_id,
          url: entry.response.url as string,
          title:
            typeof entry.response.title === "string"
              ? (entry.response.title as string)
              : null,
          markdown:
            typeof entry.response.markdown === "string"
              ? (entry.response.markdown as string)
              : "",
          fetch_time_ms:
            typeof entry.response.processing_time_ms === "number"
              ? (entry.response.processing_time_ms as number)
              : 0,
          success:
            typeof entry.response.success === "boolean"
              ? (entry.response.success as boolean)
              : true,
        }
      : null;

  return (
    <CrawlInputOutput
      mode="history"
      url={url}
      waitForJs={waitForJs}
      chromeCdpUrl={chromeCdpUrl}
      result={result}
      loading={false}
      error={entry.status === "success" ? null : "Request failed"}
      requestPayload={entry.request}
      responsePayload={entry.response}
    />
  );
}

function renderSpeech(entry: TaskHistoryEntry) {
  const transcription =
    typeof entry.response.text === "string" ? (entry.response.text as string) : "";
  const segments = Array.isArray(entry.response.segments)
    ? (entry.response.segments as Array<Record<string, unknown>>)
    : [];
  const language =
    typeof entry.response.language === "string" ? (entry.response.language as string) : undefined;
  const model =
    typeof entry.response.model === "string"
      ? (entry.response.model as string)
      : typeof entry.request.model === "string"
      ? (entry.request.model as string)
      : "";
  const processingTime =
    typeof entry.response.processing_time_ms === "number"
      ? (entry.response.processing_time_ms as number)
      : undefined;

  return (
    <TryLayout
      input={{
        title: "Input",
        description: "Audio request parameters",
        children: (
          <div className="space-y-2">
            {model && (
              <div>
                <Label>Model</Label>
                <p className="text-sm font-mono bg-muted p-2 rounded">{model}</p>
              </div>
            )}
            {language && (
              <div className="text-xs text-muted-foreground">
                <span className="font-semibold">Language:</span> {language}
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              Audio data is not stored in history. Only metadata is shown.
            </p>
          </div>
        ),
        rawPayload: {
          label: "View Raw Request",
          payload: entry.request,
        },
      }}
      output={{
        title: "Output",
        description: "Transcription",
        children: (
          <div className="space-y-4">
            {transcription ? (
              <div className="space-y-2">
                <Label>Transcription</Label>
                <Textarea value={transcription} readOnly className="min-h-[200px] font-mono text-sm" />
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No transcription text recorded.</p>
            )}

            {segments.length > 0 && (
              <div className="space-y-2">
                <Label>Speaker Segments</Label>
                <div className="border rounded-lg max-h-64 overflow-y-auto divide-y">
                  {segments.map((segment, idx) => {
                    const speaker =
                      typeof segment.speaker === "string"
                        ? segment.speaker
                        : `Speaker ${idx + 1}`;
                    const start =
                      typeof segment.start === "number" ? segment.start.toFixed(1) : "0.0";
                    const end = typeof segment.end === "number" ? segment.end.toFixed(1) : "0.0";
                    return (
                      <div key={idx} className="p-2 flex items-center justify-between text-sm">
                        <span className="font-medium">{speaker}</span>
                        <span className="text-xs text-muted-foreground">
                          {start}s - {end}s
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
              {processingTime !== undefined && (
                <span>
                  <span className="font-semibold">Processing:</span> {processingTime}ms
                </span>
              )}
            </div>
          </div>
        ),
        rawPayload: {
          label: "View Raw Response",
          payload: entry.response,
        },
      }}
    />
  );
}

function renderImageOCR(entry: TaskHistoryEntry) {
  const text = typeof entry.response.text === "string" ? (entry.response.text as string) : "";
  const model =
    typeof entry.response.model === "string"
      ? (entry.response.model as string)
      : typeof entry.request.model === "string"
      ? (entry.request.model as string)
      : "";
  const outputFormat =
    typeof entry.response.output_format === "string"
      ? (entry.response.output_format as string)
      : typeof entry.request.output_format === "string"
      ? (entry.request.output_format as string)
      : "text";
  const processingTime =
    typeof entry.response.processing_time_ms === "number"
      ? (entry.response.processing_time_ms as number)
      : undefined;

  return (
    <TryLayout
      input={{
        title: "Input",
        description: "OCR request parameters",
        children: (
          <div className="space-y-2">
            {model && (
              <div>
                <Label>Model</Label>
                <p className="text-sm font-mono bg-muted p-2 rounded">{model}</p>
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              Images are not stored in history to conserve space.
            </p>
          </div>
        ),
        rawPayload: {
          label: "View Raw Request",
          payload: entry.request,
        },
      }}
      output={{
        title: "Output",
        description: "Extracted text",
        children: (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Format: {outputFormat}</span>
              {processingTime !== undefined && <span>{processingTime}ms</span>}
            </div>
            <Textarea
              value={text}
              readOnly
              className="min-h-[200px] font-mono text-sm"
              placeholder="No text captured"
            />
          </div>
        ),
        rawPayload: {
          label: "View Raw Response",
          payload: entry.response,
        },
      }}
    />
  );
}

const TASK_RENDERERS: Record<string, TaskRenderer> = {
  "text-generation": renderTextGeneration,
  "image-captioning": renderImageCaption,
  "text-to-embedding": renderEmbeddings,
  crawl: renderCrawl,
  "automatic-speech-recognition": renderSpeech,
  "image-ocr": renderImageOCR,
};

function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

export function TaskHistoryList() {
  const [history, setHistory] = useState<TaskHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("all");

  const loadHistory = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/history/all?limit=50");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setHistory(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const filteredHistory = useMemo(() => {
    if (filter === "all") {
      return history;
    }
    return history.filter((entry) => entry.service === filter);
  }, [filter, history]);

  const toggleEntry = (requestId: string) => {
    setExpandedId((prev) => (prev === requestId ? null : requestId));
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle>Task History</CardTitle>
            <CardDescription>Latest activity across all services</CardDescription>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <Select value={filter} onValueChange={setFilter}>
              <SelectTrigger className="sm:w-48">
                <SelectValue placeholder="Filter by task" />
              </SelectTrigger>
              <SelectContent>
                {FILTER_OPTIONS.map(({ value, label }) => (
                  <SelectItem key={value} value={value}>
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" onClick={loadHistory} disabled={loading}>
              <RefreshCcw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {loading && (
          <div className="space-y-3">
            {[...Array(3)].map((_, index) => (
              <div key={`skeleton-${index}`} className="rounded-lg border p-4 space-y-2">
                <Skeleton className="h-4 w-1/3" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-2/3" />
              </div>
            ))}
          </div>
        )}

        {error && <div className="text-destructive text-sm">{error}</div>}

        {!loading && !error && filteredHistory.length === 0 && (
          <p className="text-muted-foreground text-sm">
            No history yet for {filter === "all" ? "any task" : SERVICE_LABELS[filter] ?? filter}.
          </p>
        )}

        {!loading && !error && filteredHistory.length > 0 && (
          <ScrollArea className="max-h-[640px] pr-4">
            <div className="space-y-3">
              {filteredHistory.map((entry) => {
                const isExpanded = expandedId === entry.request_id;
                const renderer = TASK_RENDERERS[entry.service];
                return (
                  <Collapsible
                    key={entry.request_id}
                    open={isExpanded}
                    onOpenChange={() => toggleEntry(entry.request_id)}
                  >
                    <div className="rounded-lg border">
                      <CollapsibleTrigger asChild>
                        <div className="p-3 hover:bg-accent/50 cursor-pointer transition-colors">
                          <div className="flex items-center justify-between gap-4">
                            <div className="flex flex-col gap-1">
                              <div className="flex items-center gap-2">
                                <Badge variant="secondary">
                                  {SERVICE_LABELS[entry.service] ?? entry.service}
                                </Badge>
                                <Badge
                                  variant={entry.status === "success" ? "default" : "destructive"}
                                  className="text-xs"
                                >
                                  {entry.status}
                                </Badge>
                              </div>
                              <span className="text-xs text-muted-foreground font-mono">
                                {entry.request_id}
                              </span>
                            </div>
                            <div className="flex items-center gap-4 text-sm text-muted-foreground">
                              {typeof entry.response.processing_time_ms === "number" && (
                                <span>{entry.response.processing_time_ms}ms</span>
                              )}
                              <span>{formatTimestamp(entry.timestamp)}</span>
                              {isExpanded ? (
                                <ChevronUp className="h-4 w-4" />
                              ) : (
                                <ChevronDown className="h-4 w-4" />
                              )}
                            </div>
                          </div>
                        </div>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="border-t p-4">
                          {renderer ? (
                            renderer(entry)
                          ) : (
                            <div className="space-y-3">
                              <p className="text-sm text-muted-foreground">
                                No renderer available for {entry.service}. Showing raw data.
                              </p>
                              <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto">
                                {JSON.stringify(entry, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>
                );
              })}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
