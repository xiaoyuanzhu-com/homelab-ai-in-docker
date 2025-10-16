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

export default function Home() {
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
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
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-6xl mx-auto px-4 py-16">
        {/* Header */}
        <header className="text-center mb-16">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Homelab AI Services
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            REST API wrapping common AI capabilities for homelab developers
          </p>
        </header>

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
