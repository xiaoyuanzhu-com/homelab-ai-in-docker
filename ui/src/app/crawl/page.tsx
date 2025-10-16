"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { HistoryPanel } from "@/components/history-panel";

interface CrawlResult {
  request_id: string;
  url: string;
  title: string | null;
  markdown: string;
  fetch_time_ms: number;
  success: boolean;
}

export default function CrawlPage() {
  const [url, setUrl] = useState("");
  const [waitForJs, setWaitForJs] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CrawlResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCrawl = async () => {
    if (!url) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/crawl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          screenshot: false,
          wait_for_js: waitForJs,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to crawl URL");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Web Crawl</h1>
        <p className="text-muted-foreground">
          Extract clean content from any URL with JavaScript rendering support
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Input */}
        <Card>
          <CardHeader>
            <CardTitle>Input</CardTitle>
            <CardDescription>Configure your crawl request</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="url">URL</Label>
              <Input
                id="url"
                type="url"
                placeholder="https://example.com"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="waitForJs"
                checked={waitForJs}
                onChange={(e) => setWaitForJs(e.target.checked)}
                className="w-4 h-4"
              />
              <Label htmlFor="waitForJs">Wait for JavaScript</Label>
            </div>

            <Button onClick={handleCrawl} disabled={loading || !url} className="w-full">
              {loading ? "Crawling..." : "Crawl URL"}
            </Button>
          </CardContent>
        </Card>

        {/* Right: Output */}
        <Card>
          <CardHeader>
            <CardTitle>Output</CardTitle>
            <CardDescription>Crawl results</CardDescription>
          </CardHeader>
          <CardContent>
            {loading && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                Crawling...
              </div>
            )}

            {error && (
              <div className="text-destructive">
                <strong>Error:</strong> {error}
              </div>
            )}

            {result && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Badge variant={result.success ? "default" : "destructive"}>
                    {result.success ? "Success" : "Failed"}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    {result.fetch_time_ms}ms
                  </span>
                </div>

                {result.title && (
                  <div>
                    <Label>Title</Label>
                    <p className="text-sm mt-1">{result.title}</p>
                  </div>
                )}

                <div>
                  <Label>Markdown Content</Label>
                  <div className="mt-2 p-4 bg-muted rounded-lg max-h-96 overflow-y-auto">
                    <pre className="text-sm whitespace-pre-wrap">{result.markdown}</pre>
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
                Enter a URL and click "Crawl URL" to see results
              </p>
            )}
          </CardContent>
        </Card>

        {/* History Panel */}
        <HistoryPanel service="crawl" onSelectEntry={(entry) => {
          if (entry.request.url) {
            setUrl(entry.request.url);
          }
        }} />
      </div>
    </div>
  );
}
