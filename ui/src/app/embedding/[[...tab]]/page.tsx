"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { HistoryPanel } from "@/components/history-panel";

interface EmbeddingResult {
  request_id: string;
  embeddings: number[][];
  dimensions: number;
  model_used: string;
  processing_time_ms: number;
}

export default function EmbeddingPage() {
  const router = useRouter();
  const params = useParams();
  const tab = (params.tab as string[]) || [];
  const [activeTab, setActiveTab] = useState(tab[0] || "try");

  const [texts, setTexts] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EmbeddingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");

  useEffect(() => {
    // Infer API base URL from current window location
    if (typeof window !== "undefined") {
      setApiBaseUrl(window.location.origin);
    }
  }, []);

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    const path = tab === "try" ? "/embedding" : `/embedding/${tab}`;
    router.push(path, { scroll: false });
  };

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
      toast.success("Embeddings generated successfully!", {
        description: `Processed ${data.embeddings.length} text(s) in ${data.processing_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to generate embeddings";
      setError(errorMsg);
      toast.error("Embedding generation failed", {
        description: errorMsg,
      });
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

      <Tabs value={activeTab} onValueChange={handleTabChange}>
        <TabsList className="mb-6">
          <TabsTrigger value="try">Try</TabsTrigger>
          <TabsTrigger value="doc">Doc</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Try Tab */}
        <TabsContent value="try">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && e.ctrlKey) {
                        handleEmbed();
                      }
                    }}
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
                    Enter texts and click "Generate Embeddings" to see results
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* API Tab */}
        <TabsContent value="doc">
          <Card>
            <CardHeader>
              <CardTitle>API Reference</CardTitle>
              <CardDescription>HTTP endpoint details and examples</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">Endpoint</h3>
                <div className="bg-muted p-4 rounded-lg">
                  <code className="text-sm">POST /api/embed</code>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Request Body</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "texts": [
    "The quick brown fox",
    "jumps over the lazy dog"
  ],
  "model": "all-MiniLM-L6-v2"
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Parameters</h3>
                <ul className="space-y-2 text-sm">
                  <li><code className="bg-muted px-2 py-1 rounded">texts</code> (array[string], required) - List of texts to embed</li>
                  <li><code className="bg-muted px-2 py-1 rounded">model</code> (string, optional) - Model name (default: all-MiniLM-L6-v2)</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">Response</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`{
  "request_id": "uuid",
  "embeddings": [
    [0.1234, -0.5678, ...],
    [0.9012, -0.3456, ...]
  ],
  "dimensions": 384,
  "model_used": "all-MiniLM-L6-v2",
  "processing_time_ms": 45
}`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">cURL Example</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`curl -X POST ${apiBaseUrl}/api/embed \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": ["Hello world", "Semantic search"]
  }'`}
                </pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history">
          <HistoryPanel
            service="embed"
            onSelectEntry={(entry) => {
              if (entry.request.texts) {
                setTexts(entry.request.texts.join("\n"));
                setActiveTab("try");
                router.push("/embedding", { scroll: false });
              }
            }}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
