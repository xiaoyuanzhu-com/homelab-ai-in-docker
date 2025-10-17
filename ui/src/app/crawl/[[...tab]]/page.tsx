"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { ChevronDown, AlertCircle, Info } from "lucide-react";
import { toast } from "sonner";
import { HistoryPanel } from "@/components/history-panel";
import { MarkdownDoc } from "@/components/markdown-doc";
import crawlDoc from "@/docs/crawl.md";

interface CrawlResult {
  request_id: string;
  url: string;
  title: string | null;
  markdown: string;
  fetch_time_ms: number;
  success: boolean;
}

export default function CrawlPage() {
  const router = useRouter();
  const params = useParams();
  const tab = (params.tab as string[]) || [];
  const [activeTab, setActiveTab] = useState(tab[0] || "try");
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [url, setUrl] = useState("");
  const [waitForJs, setWaitForJs] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CrawlResult | null>(null);
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
    const path = tab === "try" ? "/crawl" : `/crawl/${tab}`;
    router.push(path, { scroll: false });
  };

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
      toast.success("Crawl completed successfully!", {
        description: `Processed in ${data.fetch_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to crawl URL";
      setError(errorMsg);
      toast.error("Crawl failed", {
        description: errorMsg,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <TooltipProvider>
      <div className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Web Crawl</h1>
          <p className="text-muted-foreground">
            Extract clean content from any URL with JavaScript rendering support
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
                      onKeyDown={(e) => e.key === "Enter" && handleCrawl()}
                    />
                  </div>

                  {/* Advanced Options - Collapsible */}
                  <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
                    <CollapsibleTrigger asChild>
                      <Button variant="ghost" className="w-full justify-between p-0 h-auto font-normal">
                        <span className="text-sm text-muted-foreground">Advanced options</span>
                        <ChevronDown className={`h-4 w-4 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="space-y-4 pt-4">
                      <div className="flex items-center justify-between space-x-2">
                        <div className="flex items-center gap-2">
                          <Label htmlFor="waitForJs" className="text-sm">
                            Wait for JavaScript
                          </Label>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Info className="h-4 w-4 text-muted-foreground cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>Waits 2 seconds for dynamic content to load.</p>
                              <p className="text-xs text-muted-foreground">Enable for React, Vue, or SPA sites.</p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <Switch
                          id="waitForJs"
                          checked={waitForJs}
                          onCheckedChange={setWaitForJs}
                        />
                      </div>
                    </CollapsibleContent>
                  </Collapsible>

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

                  {result && !loading && (
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
            </div>
          </TabsContent>

          {/* Doc Tab */}
          <TabsContent value="doc">
            <Card>
              <CardContent className="pt-6">
                <MarkdownDoc content={crawlDoc} apiBaseUrl={apiBaseUrl} />
              </CardContent>
            </Card>
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history">
            <HistoryPanel
              service="crawl"
              onSelectEntry={(entry) => {
                if (entry.request.url) {
                  setUrl(entry.request.url);
                  setActiveTab("try");
                  router.push("/crawl", { scroll: false });
                }
              }}
            />
          </TabsContent>
        </Tabs>
      </div>
    </TooltipProvider>
  );
}
