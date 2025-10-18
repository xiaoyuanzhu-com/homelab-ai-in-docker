"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { AlertCircle, ChevronDown, ChevronUp, Info } from "lucide-react";
import { MarkdownDoc } from "@/components/markdown-doc";

interface CrawlResult {
  request_id: string;
  url: string;
  title: string | null;
  markdown: string;
  fetch_time_ms: number;
  success: boolean;
}

interface CrawlInputOutputProps {
  mode: "try" | "history";

  // Input state
  url: string;
  onUrlChange?: (url: string) => void;
  waitForJs: boolean;
  onWaitForJsChange?: (value: boolean) => void;

  // Output/Result
  result: CrawlResult | null;
  loading: boolean;
  error: string | null;

  // Actions
  onSend?: () => void;
}

export function CrawlInputOutput({
  mode,
  url,
  onUrlChange,
  waitForJs,
  onWaitForJsChange,
  result,
  loading,
  error,
  onSend,
}: CrawlInputOutputProps) {
  const isEditable = mode === "try";
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [isInputOpen, setIsInputOpen] = useState(false);
  const [isOutputOpen, setIsOutputOpen] = useState(false);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && isEditable && onSend) {
      onSend();
    }
  };

  return (
    <TooltipProvider>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Input */}
        <Card>
          <CardHeader>
            <CardTitle>Input</CardTitle>
            <CardDescription>
              {isEditable ? "Enter URL to crawl" : "Input URL"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
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

            <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="outline" size="sm" className="w-full justify-between">
                  <span>Advanced Options</span>
                  <ChevronDown className="h-4 w-4" />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="pt-4 space-y-4">
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
                          Wait for JavaScript to execute before capturing content.
                          Useful for dynamic sites.
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
              </CollapsibleContent>
            </Collapsible>

            {isEditable && onSend && (
              <Button onClick={onSend} disabled={loading || !url.trim()} className="w-full">
                {loading ? "Crawling..." : "Crawl URL"}
              </Button>
            )}

            <Collapsible open={isInputOpen} onOpenChange={setIsInputOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="w-full justify-between">
                  <span className="text-sm font-medium">View Raw Request</span>
                  {isInputOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="pt-2">
                <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto">
                  {JSON.stringify(
                    {
                      url,
                      wait_for_js: waitForJs,
                    },
                    null,
                    2
                  )}
                </pre>
              </CollapsibleContent>
            </Collapsible>
          </CardContent>
        </Card>

        {/* Right: Output */}
        <Card>
          <CardHeader>
            <CardTitle>Output</CardTitle>
            <CardDescription>Crawled content</CardDescription>
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
                  <div className="mt-2 border rounded-lg">
                    <MarkdownDoc content={result.markdown} />
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
                {isEditable
                  ? 'Enter a URL and click "Crawl URL" to see results'
                  : "No output available"}
              </p>
            )}

            {result && (
              <Collapsible open={isOutputOpen} onOpenChange={setIsOutputOpen} className="mt-4">
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="w-full justify-between">
                    <span className="text-sm font-medium">View Raw Response</span>
                    {isOutputOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="pt-2">
                  <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto max-h-96">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                </CollapsibleContent>
              </Collapsible>
            )}
          </CardContent>
        </Card>
      </div>
    </TooltipProvider>
  );
}
