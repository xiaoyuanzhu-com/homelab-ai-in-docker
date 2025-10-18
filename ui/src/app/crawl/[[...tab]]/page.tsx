"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import { toast } from "sonner";
import { MarkdownDoc } from "@/components/markdown-doc";
import { CrawlInputOutput } from "@/components/crawl-input-output";
import { CrawlHistory } from "@/components/crawl-history";

const crawlDoc = `# API Reference

HTTP endpoint details and examples

## Endpoint

\`\`\`
POST /api/crawl
\`\`\`

## Request Body

\`\`\`json
{
  "url": "https://example.com",
  "screenshot": false,
  "wait_for_js": true,
  "chrome_cdp_url": "http://172.16.2.2:9223"
}
\`\`\`

## Parameters

- \`url\` (string, required) - The URL to crawl
- \`screenshot\` (boolean, optional) - Capture screenshot (default: false)
- \`wait_for_js\` (boolean, optional) - Wait for JavaScript execution (default: true)
- \`chrome_cdp_url\` (string, optional) - Remote Chrome CDP URL for browser connection

## Response

\`\`\`json
{
  "request_id": "uuid",
  "url": "https://example.com/",
  "title": "Example Domain",
  "markdown": "# Example Domain\\n\\nThis domain...",
  "html": "<!DOCTYPE html>...",
  "screenshot_base64": null,
  "fetch_time_ms": 1580,
  "success": true
}
\`\`\`

## cURL Example

\`\`\`bash
curl -X POST {{API_BASE_URL}}/api/crawl \\
  -H "Content-Type: application/json" \\
  -d '{
    "url": "https://example.com",
    "wait_for_js": true,
    "chrome_cdp_url": "http://172.16.2.2:9223"
  }'
\`\`\`
`;

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

  const [url, setUrl] = useState("");
  const [waitForJs, setWaitForJs] = useState(true);
  const [chromeCdpUrl, setChromeCdpUrl] = useState("");
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
      const requestBody: any = {
        url,
        screenshot: false,
        wait_for_js: waitForJs,
      };

      // Only include chrome_cdp_url if it's not empty
      if (chromeCdpUrl.trim()) {
        requestBody.chrome_cdp_url = chromeCdpUrl.trim();
      }

      const response = await fetch("/api/crawl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
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
            <CrawlInputOutput
              mode="try"
              url={url}
              onUrlChange={setUrl}
              waitForJs={waitForJs}
              onWaitForJsChange={setWaitForJs}
              chromeCdpUrl={chromeCdpUrl}
              onChromeCdpUrlChange={setChromeCdpUrl}
              result={result}
              loading={loading}
              error={error}
              onSend={handleCrawl}
            />
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
            <CrawlHistory />
          </TabsContent>
        </Tabs>
      </div>
    </TooltipProvider>
  );
}
