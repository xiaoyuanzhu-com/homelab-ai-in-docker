"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface HistoryEntry {
  timestamp: string;
  request_id: string;
  status: string;
  request: Record<string, unknown>;
  response: Record<string, unknown>;
}

interface HistoryPanelProps {
  service: string;
  onSelectEntry?: (entry: HistoryEntry) => void;
}

export function HistoryPanel({ service, onSelectEntry }: HistoryPanelProps) {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/history/${service}?limit=20`);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [service]);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>History</CardTitle>
            <CardDescription>Recent {service} requests</CardDescription>
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

        {error && (
          <div className="text-destructive text-sm">{error}</div>
        )}

        {!loading && !error && history.length === 0 && (
          <p className="text-muted-foreground text-sm">No history yet</p>
        )}

        {!loading && !error && history.length > 0 && (
          <ScrollArea className="h-[400px] pr-4">
            <div className="space-y-2">
              {history.map((entry) => (
                <div
                  key={entry.request_id}
                  className="p-3 rounded-lg border hover:bg-accent/50 cursor-pointer transition-colors"
                  onClick={() => onSelectEntry?.(entry)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant={entry.status === "success" ? "default" : "destructive"} className="text-xs">
                      {entry.status}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {formatTimestamp(entry.timestamp)}
                    </span>
                  </div>
                  <div className="text-sm">
                    {typeof entry.response.processing_time_ms === "number" && (
                      <span className="text-muted-foreground">
                        {entry.response.processing_time_ms}ms
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground font-mono mt-1 truncate">
                    {entry.request_id}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
