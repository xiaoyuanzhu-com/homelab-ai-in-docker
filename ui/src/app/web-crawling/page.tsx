"use client";

import { useState } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { toast } from "sonner";
import { CrawlInputOutput } from "@/components/crawl-input-output";
import { CrawlResult } from "@/types/api";

export default function CrawlPage() {
  const [url, setUrl] = useState("");
  const [waitForJs, setWaitForJs] = useState(true);
  const [chromeCdpUrl, setChromeCdpUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CrawlResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCrawl = async () => {
    if (!url) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const requestBody: {
        url: string;
        screenshot: boolean;
        wait_for_js: boolean;
        chrome_cdp_url?: string;
      } = {
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
        description: `Processed in ${data.processing_time_ms}ms`,
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
          <h1 className="text-3xl font-bold mb-2">Web Crawling</h1>
          <p className="text-muted-foreground">
            Extract clean content from any URL with JavaScript rendering support
          </p>
        </div>

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
      </div>
    </TooltipProvider>
  );
}
