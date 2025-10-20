"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";

export default function SettingsPage() {
  const [chromeCdpUrl, setChromeCdpUrl] = useState("");
  const [maxConcurrentTasks, setMaxConcurrentTasks] = useState("5");
  const [enableGpu, setEnableGpu] = useState(true);
  const [cacheResults, setCacheResults] = useState(true);
  const [logLevel, setLogLevel] = useState("info");

  const handleSaveGeneral = () => {
    // In a real implementation, this would call the API to save settings
    toast.success("Settings saved successfully!");
  };

  const handleSaveCrawler = () => {
    toast.success("Crawler settings saved successfully!");
  };

  const handleSavePerformance = () => {
    toast.success("Performance settings saved successfully!");
  };

  const handleClearCache = () => {
    toast.success("Cache cleared successfully!");
  };

  const handleClearHistory = () => {
    toast.success("History cleared successfully!");
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Settings</h1>
        <p className="text-muted-foreground">
          Configure application behavior and preferences
        </p>
      </div>

      <Tabs defaultValue="general" className="w-full">
        <TabsList className="mb-6">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="crawler">Web Crawler</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
        </TabsList>

        {/* General Settings */}
        <TabsContent value="general">
          <Card>
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
              <CardDescription>Basic application configuration</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="log-level">Log Level</Label>
                <select
                  id="log-level"
                  value={logLevel}
                  onChange={(e) => setLogLevel(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md bg-background"
                >
                  <option value="debug">Debug</option>
                  <option value="info">Info</option>
                  <option value="warning">Warning</option>
                  <option value="error">Error</option>
                </select>
                <p className="text-sm text-muted-foreground">
                  Control the verbosity of application logs
                </p>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label>Cache Results</Label>
                  <p className="text-sm text-muted-foreground">
                    Cache API responses to improve performance
                  </p>
                </div>
                <Switch checked={cacheResults} onCheckedChange={setCacheResults} />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label>Enable GPU Acceleration</Label>
                  <p className="text-sm text-muted-foreground">
                    Use GPU for AI model inference when available
                  </p>
                </div>
                <Switch checked={enableGpu} onCheckedChange={setEnableGpu} />
              </div>

              <Button onClick={handleSaveGeneral}>Save General Settings</Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Crawler Settings */}
        <TabsContent value="crawler">
          <Card>
            <CardHeader>
              <CardTitle>Web Crawler Settings</CardTitle>
              <CardDescription>Configure web scraping behavior</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="chrome-cdp">Chrome CDP URL</Label>
                <Input
                  id="chrome-cdp"
                  type="text"
                  placeholder="http://chrome:9222"
                  value={chromeCdpUrl}
                  onChange={(e) => setChromeCdpUrl(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">
                  Remote Chrome DevTools Protocol endpoint for distributed crawling
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="max-concurrent">Max Concurrent Requests</Label>
                <Input
                  id="max-concurrent"
                  type="number"
                  min="1"
                  max="20"
                  value={maxConcurrentTasks}
                  onChange={(e) => setMaxConcurrentTasks(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">
                  Maximum number of simultaneous crawl requests
                </p>
              </div>

              <Button onClick={handleSaveCrawler}>Save Crawler Settings</Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Settings */}
        <TabsContent value="performance">
          <Card>
            <CardHeader>
              <CardTitle>Performance Settings</CardTitle>
              <CardDescription>Optimize resource usage and speed</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Model Loading</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Models are loaded on-demand to save memory. First requests may take longer.
                  </p>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">Preload All Models</Button>
                    <Button variant="outline" size="sm">Unload All Models</Button>
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Request Queuing</h3>
                  <p className="text-sm text-muted-foreground">
                    Requests are queued when system resources are limited. Adjust concurrency limits
                    in the crawler settings.
                  </p>
                </div>
              </div>

              <Button onClick={handleSavePerformance}>Save Performance Settings</Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Data Management */}
        <TabsContent value="data">
          <Card>
            <CardHeader>
              <CardTitle>Data Management</CardTitle>
              <CardDescription>Manage cached data and history</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Cache</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Cached API responses to improve performance. Clearing cache will not affect
                    saved models.
                  </p>
                  <Button variant="destructive" size="sm" onClick={handleClearCache}>
                    Clear Cache
                  </Button>
                </div>

                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Request History</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Historical records of all API requests. Clearing history will permanently delete
                    all request logs.
                  </p>
                  <Button variant="destructive" size="sm" onClick={handleClearHistory}>
                    Clear History
                  </Button>
                </div>

                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Downloaded Models</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    AI models are stored locally. Manage models from the Models page.
                  </p>
                  <Button variant="outline" size="sm" asChild>
                    <a href="/models">Go to Models</a>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
