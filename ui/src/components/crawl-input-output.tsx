"use client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Textarea } from "@/components/ui/textarea";
import { AlertCircle, Info } from "lucide-react";
import { TryLayout } from "@/components/try/try-layout";
import { CrawlResult } from "@/types/api";

interface CrawlInputOutputProps {
  mode: "try" | "history";

  // Input state
  url: string;
  onUrlChange?: (url: string) => void;
  waitForJs: boolean;
  onWaitForJsChange?: (value: boolean) => void;
  chromeCdpUrl?: string;
  onChromeCdpUrlChange?: (url: string) => void;

  // Output/Result
  result: CrawlResult | null;
  loading: boolean;
  error: string | null;
  responsePayload?: Record<string, unknown>;

  // Actions
  onSend?: () => void;

  // Overrides for history playback
  requestPayload?: Record<string, unknown>;
}

export function CrawlInputOutput({
  mode,
  url,
  onUrlChange,
  waitForJs,
  onWaitForJsChange,
  chromeCdpUrl,
  onChromeCdpUrlChange,
  result,
  loading,
  error,
  onSend,
  requestPayload,
  responsePayload,
}: CrawlInputOutputProps) {
  const isEditable = mode === "try";

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && isEditable && onSend) {
      onSend();
    }
  };

  const rawRequestData =
    requestPayload ??
    {
      url,
      wait_for_js: waitForJs,
      ...(chromeCdpUrl && { chrome_cdp_url: chromeCdpUrl }),
    };

  const rawResponseData = responsePayload ?? result;

  const inputContent = (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="url">URL</Label>
        <Input
          id="url"
          type="url"
          placeholder="https://example.com"
          value={url}
          onChange={(e) => onUrlChange?.(e.target.value)}
          onKeyDown={handleKeyDown}
          readOnly={!isEditable}
          className={!isEditable ? "bg-muted" : ""}
        />
      </div>

      <div className="flex items-center justify-between space-x-2">
        <div className="flex items-center space-x-2">
          <Label htmlFor="wait-for-js" className="cursor-pointer">
            Wait for JavaScript
          </Label>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-4 w-4 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>
              <p className="max-w-xs">
                Wait for JavaScript to execute before capturing content. Useful for dynamic sites.
              </p>
            </TooltipContent>
          </Tooltip>
        </div>
        <Switch
          id="wait-for-js"
          checked={waitForJs}
          onCheckedChange={onWaitForJsChange}
          disabled={!isEditable}
        />
      </div>

      <div className="space-y-2">
        <div className="flex items-center space-x-2">
          <Label htmlFor="chrome-cdp-url">Remote Chrome URL</Label>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-4 w-4 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>
              <p className="max-w-xs">
                Optional Chrome DevTools Protocol (CDP) URL for remote browser. Example: http://172.16.2.2:9223
              </p>
            </TooltipContent>
          </Tooltip>
        </div>
        <Input
          id="chrome-cdp-url"
          type="url"
          placeholder="http://172.16.2.2:9223 (optional)"
          value={chromeCdpUrl || ""}
          onChange={(e) => onChromeCdpUrlChange?.(e.target.value)}
          readOnly={!isEditable}
          className={!isEditable ? "bg-muted" : ""}
        />
      </div>
    </div>
  );

  const outputContent = (
    <>
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

      {result && !error && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant={result.success ? "default" : "destructive"}>
              {result.success ? "Success" : "Failed"}
            </Badge>
            {result.processing_time_ms !== undefined && (
              <span className="text-sm text-muted-foreground">
                {result.processing_time_ms}ms
              </span>
            )}
          </div>

          {result.title && (
            <div className="space-y-2">
              <Label>Title</Label>
              <p className="text-sm">{result.title}</p>
            </div>
          )}

          <div className="space-y-2">
            <Label>Markdown Content</Label>
            <Textarea
              value={result.markdown}
              readOnly
              className="font-mono text-sm min-h-[400px] resize-y"
            />
          </div>

          <div className="space-y-2">
            <Label>Request ID</Label>
            <p className="text-xs text-muted-foreground font-mono">
              {result.request_id}
            </p>
          </div>
        </div>
      )}

      {!loading && !error && !result && (
        <p className="text-muted-foreground text-center py-8">
          {isEditable ? 'Enter a URL and click "Crawl URL" to see results' : "No output available"}
        </p>
      )}
    </>
  );

  return (
    <TooltipProvider>
      <TryLayout
        input={{
          title: "Input",
          description: isEditable ? "Enter URL to crawl" : "Input URL",
          children: inputContent,
          footer:
            isEditable && onSend ? (
              <Button onClick={onSend} disabled={loading || !url.trim()} className="w-full">
                {loading ? "Crawling..." : "Crawl URL"}
              </Button>
            ) : null,
          rawPayload: {
            label: "View Raw Request",
            payload: rawRequestData,
          },
        }}
        output={{
          title: "Output",
          description: "Crawled content",
          children: outputContent,
          rawPayload: {
            label: "View Raw Response",
            payload: rawResponseData,
          },
        }}
      />
    </TooltipProvider>
  );
}
