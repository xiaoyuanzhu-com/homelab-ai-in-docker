"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { HistoryPanel } from "@/components/history-panel";

interface EmbeddingResult {
  request_id: string;
  embeddings: number[][];
  dimensions: number;
  model_used: string;
  processing_time_ms: number;
}

export default function EmbeddingPage() {
  const [texts, setTexts] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EmbeddingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleEmbed = async () => {
    const textList = texts.split("\n").filter((t) => t.trim());
    if (textList.length === 0) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/embed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: textList }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate embeddings");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Text Embedding</h1>
        <p className="text-muted-foreground">
          Convert text into vector representations for semantic search and similarity matching
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Input */}
        <Card>
          <CardHeader>
            <CardTitle>Input</CardTitle>
            <CardDescription>Enter texts to embed (one per line)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="texts">Texts</Label>
              <Textarea
                id="texts"
                placeholder="Enter text here&#10;One text per line&#10;Example: The quick brown fox"
                value={texts}
                onChange={(e) => setTexts(e.target.value)}
                rows={10}
              />
              <p className="text-xs text-muted-foreground">
                {texts.split("\n").filter((t) => t.trim()).length} text(s)
              </p>
            </div>

            <Button onClick={handleEmbed} disabled={loading || !texts.trim()} className="w-full">
              {loading ? "Generating..." : "Generate Embeddings"}
            </Button>
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
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                Generating embeddings...
              </div>
            )}

            {error && (
              <div className="text-destructive">
                <strong>Error:</strong> {error}
              </div>
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
                Enter texts and click "Generate Embeddings" to see results
              </p>
            )}
          </CardContent>
        </Card>

        {/* History Panel */}
        <HistoryPanel service="embed" onSelectEntry={(entry) => {
          if (entry.request.texts) {
            setTexts(entry.request.texts.join("\n"));
          }
        }} />
      </div>
    </div>
  );
}
