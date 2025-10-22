"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronUp } from "lucide-react";
import { CrawlInputOutput } from "@/components/crawl-input-output";

interface HistoryEntry {
  service: string;
  timestamp: string;
  request_id: string;
  status: string;
  request: Record<string, unknown>;
  response: Record<string, unknown>;
}

interface CrawlResult {
  request_id: string;
  url: string;
  title: string | null;
  markdown: string;
  fetch_time_ms: number;
  success: boolean;
}

export function CrawlHistory() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/history/crawl?limit=20");
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

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const toggleEntry = (requestId: string) => {
    setExpandedId(expandedId === requestId ? null : requestId);
  };

  const extractEntryData = (entry: HistoryEntry) => {
    const url = typeof entry.request.url === "string" ? entry.request.url : "";
    const waitForJs =
      typeof entry.request.wait_for_js === "boolean" ? entry.request.wait_for_js : true;

    const result: CrawlResult | null =
      entry.response.url && typeof entry.response.url === "string"
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
              typeof entry.response.fetch_time_ms === "number"
                ? (entry.response.fetch_time_ms as number)
                : 0,
            success:
              typeof entry.response.success === "boolean"
                ? (entry.response.success as boolean)
                : false,
          }
        : null;

    return { url, waitForJs, result };
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>History</CardTitle>
            <CardDescription>Recent crawl requests</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={loadHistory}>
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
            Loading history...
          </div>
        )}

        {error && <div className="text-destructive text-sm">{error}</div>}

        {!loading && !error && history.length === 0 && (
          <p className="text-muted-foreground text-sm">No history yet</p>
        )}

        {!loading && !error && history.length > 0 && (
          <ScrollArea className="h-[600px] pr-4">
            <div className="space-y-3">
              {history.map((entry) => {
                const isExpanded = expandedId === entry.request_id;
                const { url, waitForJs, result } = extractEntryData(entry);

                return (
                  <Collapsible
                    key={entry.request_id}
                    open={isExpanded}
                    onOpenChange={() => toggleEntry(entry.request_id)}
                  >
                    <div className="rounded-lg border">
                      {/* Header - Always Visible */}
                      <CollapsibleTrigger asChild>
                        <div className="p-3 hover:bg-accent/50 cursor-pointer transition-colors">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Badge
                                variant={
                                  entry.status === "success" ? "default" : "destructive"
                                }
                                className="text-xs"
                              >
                                {entry.status}
                              </Badge>
                              <span className="text-xs text-muted-foreground">
                                {formatTimestamp(entry.timestamp)}
                              </span>
                            </div>
                            {isExpanded ? (
                              <ChevronUp className="h-4 w-4 text-muted-foreground" />
                            ) : (
                              <ChevronDown className="h-4 w-4 text-muted-foreground" />
                            )}
                          </div>
                          <div className="flex items-center gap-3 text-sm">
                            {typeof entry.response.fetch_time_ms === "number" && (
                              <span className="text-muted-foreground">
                                {entry.response.fetch_time_ms}ms
                              </span>
                            )}
                            <span className="text-muted-foreground truncate flex-1">
                              {url}
                            </span>
                          </div>
                          <div className="text-xs text-muted-foreground font-mono mt-1 truncate">
                            {entry.request_id}
                          </div>
                        </div>
                      </CollapsibleTrigger>

                      {/* Expanded Content */}
                      <CollapsibleContent>
                        <div className="border-t p-4">
                          <CrawlInputOutput
                            mode="history"
                            url={url}
                            waitForJs={waitForJs}
                            result={result}
                            loading={false}
                            error={null}
                          />
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
