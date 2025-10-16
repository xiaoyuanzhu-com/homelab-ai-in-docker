"use client";

import { useEffect, useState } from "react";

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
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-white">
            API Status
          </h2>
          {isLoading ? (
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
              <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              Checking API status...
            </div>
          ) : error ? (
            <div className="text-red-600 dark:text-red-400">
              ✗ API Offline: {error}
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></span>
                <span className="text-green-600 dark:text-green-400 font-medium">
                  {apiStatus?.status === "running" ? "Online" : "Unknown"}
                </span>
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Version: {apiStatus?.version}
              </div>
            </div>
          )}
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {/* Crawl Feature */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow">
            <div className="text-4xl mb-4">🌐</div>
            <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-white">
              Smart Web Scraping
            </h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Crawl URLs with JavaScript rendering support, extract clean
              Markdown content
            </p>
            <span className="inline-block px-3 py-1 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-full text-sm font-medium">
              Available
            </span>
          </div>

          {/* Embedding Feature */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow opacity-60">
            <div className="text-4xl mb-4">🔢</div>
            <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-white">
              Text Embedding
            </h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Convert text to vectors for semantic search and similarity
              matching
            </p>
            <span className="inline-block px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-full text-sm font-medium">
              Coming Soon
            </span>
          </div>

          {/* Captioning Feature */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow opacity-60">
            <div className="text-4xl mb-4">🖼️</div>
            <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-white">
              Image Captioning
            </h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Generate descriptive text from images for accessibility and
              search
            </p>
            <span className="inline-block px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-full text-sm font-medium">
              Coming Soon
            </span>
          </div>
        </div>

        {/* Quick Links */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-white">
            Quick Links
          </h2>
          <div className="grid sm:grid-cols-2 gap-4">
            <a
              href="/api/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 transition-colors"
            >
              <span className="text-2xl">📚</span>
              <div>
                <div className="font-medium text-gray-800 dark:text-white">
                  API Documentation
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Interactive Swagger UI
                </div>
              </div>
            </a>
            <a
              href="/api/health"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 transition-colors"
            >
              <span className="text-2xl">💚</span>
              <div>
                <div className="font-medium text-gray-800 dark:text-white">
                  Health Check
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  API health endpoint
                </div>
              </div>
            </a>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 text-gray-600 dark:text-gray-400">
          <p>Built for homelab developers who need AI capabilities</p>
        </footer>
      </div>
    </div>
  );
}
