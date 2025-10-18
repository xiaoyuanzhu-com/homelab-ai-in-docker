"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ApiStatus {
  name: string;
  version: string;
  status: string;
  endpoints: {
    crawl: string;
    docs: string;
    health: string;
  };
}

interface HardwareStats {
  cpu: {
    usage_percent: number;
    cores: number;
    frequency_mhz: number | null;
  };
  memory: {
    total_gb: number;
    available_gb: number;
    used_gb: number;
    usage_percent: number;
  };
  gpu: {
    available: boolean;
    count: number;
    devices: Array<{
      id: number;
      name: string;
      compute_capability: string;
      total_memory_gb: number;
      allocated_memory_gb: number;
      reserved_memory_gb: number;
      free_memory_gb: number;
      memory_usage_percent: number;
      utilization_percent?: number;
      temperature_c?: number;
    }>;
  };
  inference: {
    device: string;
    description: string;
  };
}

export default function Home() {
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null);
  const [hardwareStats, setHardwareStats] = useState<HardwareStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch API status
    fetch("/api")
      .then((res) => res.json())
      .then((data) => {
        setApiStatus(data);
        setIsLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setIsLoading(false);
      });

    // Fetch hardware stats
    fetch("/api/hardware")
      .then((res) => res.json())
      .then((data) => {
        if (!data.error) {
          setHardwareStats(data);
        }
      })
      .catch((err) => {
        console.error("Failed to fetch hardware stats:", err);
      });

    // Refresh hardware stats every 5 seconds
    const interval = setInterval(() => {
      fetch("/api/hardware")
        .then((res) => res.json())
        .then((data) => {
          if (!data.error) {
            setHardwareStats(data);
          }
        })
        .catch((err) => {
          console.error("Failed to fetch hardware stats:", err);
        });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Page Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-3 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            AI Services Dashboard
          </h1>
          <p className="text-lg text-muted-foreground">
            Monitor and manage your AI capabilities
          </p>
        </div>

        {/* API Status Card */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>API Status</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                Checking API status...
              </div>
            ) : error ? (
              <div className="text-destructive">
                ✗ API Offline: {error}
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
                  <Badge variant="outline" className="text-green-600 border-green-600">
                    {apiStatus?.status === "running" ? "Online" : "Unknown"}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  Version: {apiStatus?.version}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Hardware Stats Card */}
        {hardwareStats && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle>Hardware Resources</CardTitle>
              <CardDescription>
                {hardwareStats.inference.description}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                {/* CPU & Memory */}
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">CPU Usage</span>
                      <span className="text-sm text-muted-foreground">
                        {hardwareStats.cpu.usage_percent.toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${hardwareStats.cpu.usage_percent}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {hardwareStats.cpu.cores} cores
                      {hardwareStats.cpu.frequency_mhz &&
                        ` @ ${hardwareStats.cpu.frequency_mhz} MHz`
                      }
                    </p>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Memory Usage</span>
                      <span className="text-sm text-muted-foreground">
                        {hardwareStats.memory.used_gb.toFixed(1)} / {hardwareStats.memory.total_gb.toFixed(1)} GB
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div
                        className="bg-green-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${hardwareStats.memory.usage_percent}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {hardwareStats.memory.usage_percent.toFixed(1)}% used
                    </p>
                  </div>
                </div>

                {/* GPU */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">GPU</span>
                    <Badge variant={hardwareStats.gpu.available ? "default" : "secondary"}>
                      {hardwareStats.gpu.available ? "Available" : "Not Available"}
                    </Badge>
                  </div>
                  {hardwareStats.gpu.available && hardwareStats.gpu.devices.length > 0 ? (
                    <div className="space-y-3">
                      {hardwareStats.gpu.devices.map((gpu) => (
                        <div key={gpu.id} className="p-3 bg-muted/50 rounded-lg space-y-2">
                          <div>
                            <p className="text-sm font-medium">{gpu.name}</p>
                            <p className="text-xs text-muted-foreground">
                              Compute {gpu.compute_capability}
                            </p>
                          </div>
                          <div>
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs">VRAM</span>
                              <span className="text-xs text-muted-foreground">
                                {gpu.reserved_memory_gb.toFixed(1)} / {gpu.total_memory_gb.toFixed(1)} GB
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5 dark:bg-gray-700">
                              <div
                                className="bg-purple-600 h-1.5 rounded-full transition-all duration-300"
                                style={{ width: `${gpu.memory_usage_percent}%` }}
                              ></div>
                            </div>
                          </div>
                          {gpu.utilization_percent !== undefined && (
                            <p className="text-xs text-muted-foreground">
                              Utilization: {gpu.utilization_percent}%
                              {gpu.temperature_c !== undefined &&
                                ` • Temp: ${gpu.temperature_c}°C`
                              }
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      No GPU detected. AI models will run on CPU.
                    </p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {/* Crawl Feature */}
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="text-4xl mb-2">🌐</div>
              <CardTitle>Smart Web Scraping</CardTitle>
              <CardDescription>
                Crawl URLs with JavaScript rendering support, extract clean
                Markdown content
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Badge className="bg-green-600 hover:bg-green-700">Available</Badge>
            </CardContent>
          </Card>

          {/* Embedding Feature */}
          <Card className="hover:shadow-lg transition-shadow opacity-60">
            <CardHeader>
              <div className="text-4xl mb-2">🔢</div>
              <CardTitle>Text Embedding</CardTitle>
              <CardDescription>
                Convert text to vectors for semantic search and similarity
                matching
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Badge variant="secondary">Coming Soon</Badge>
            </CardContent>
          </Card>

          {/* Captioning Feature */}
          <Card className="hover:shadow-lg transition-shadow opacity-60">
            <CardHeader>
              <div className="text-4xl mb-2">🖼️</div>
              <CardTitle>Image Captioning</CardTitle>
              <CardDescription>
                Generate descriptive text from images for accessibility and
                search
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Badge variant="secondary">Coming Soon</Badge>
            </CardContent>
          </Card>
        </div>

        {/* Quick Links */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Links</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid sm:grid-cols-2 gap-4">
              <a
                href="/api/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 p-4 rounded-lg border hover:border-primary transition-colors"
              >
                <span className="text-2xl">📚</span>
                <div>
                  <div className="font-medium">API Documentation</div>
                  <div className="text-sm text-muted-foreground">
                    Interactive Swagger UI
                  </div>
                </div>
              </a>
              <a
                href="/api/health"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 p-4 rounded-lg border hover:border-primary transition-colors"
              >
                <span className="text-2xl">💚</span>
                <div>
                  <div className="font-medium">Health Check</div>
                  <div className="text-sm text-muted-foreground">
                    API health endpoint
                  </div>
                </div>
              </a>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <footer className="text-center mt-16 text-muted-foreground">
          <p>Built for homelab developers who need AI capabilities</p>
        </footer>
      </div>
    </div>
  );
}
