"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, Download, Trash2, Loader2, ExternalLink, X } from "lucide-react";
import { toast } from "sonner";
import { Progress } from "@/components/ui/progress";

interface EmbeddingModel {
  id: string;
  name: string;
  team: string;
  license: string;
  dimensions: number;
  languages: string;
  description: string;
  size_mb: number;
  link: string;
  is_downloaded: boolean;
  downloaded_size_mb: number | null;
}

interface ModelsTabProps {
  onModelSelect?: (modelId: string) => void;
}

interface DownloadProgress {
  percent: number;
  currentMb: number;
  totalMb: number;
}

export function ModelsTab({}: ModelsTabProps) {
  const [models, setModels] = useState<EmbeddingModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<Map<string, DownloadProgress>>(new Map());
  const [activeEventSources, setActiveEventSources] = useState<Map<string, EventSource>>(new Map());
  const [deletingModels, setDeletingModels] = useState<Set<string>>(new Set());

  const fetchModels = async () => {
    try {
      const response = await fetch("/api/models/embedding");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setModels(data.models);
      setError(null);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to load models";
      setError(errorMsg);
      toast.error("Failed to load models", {
        description: errorMsg,
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleDownload = async (modelId: string) => {
    const eventSource = new EventSource(`/api/models/embedding/${modelId}/download`);

    // Store EventSource for cancellation
    setActiveEventSources((prev) => new Map(prev).set(modelId, eventSource));

    // Initialize progress
    setDownloadProgress((prev) => new Map(prev).set(modelId, { percent: 0, currentMb: 0, totalMb: 0 }));

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "progress":
          setDownloadProgress((prev) =>
            new Map(prev).set(modelId, {
              percent: data.percent || 0,
              currentMb: data.current_mb || 0,
              totalMb: data.total_mb || 0,
            })
          );
          break;

        case "complete":
          eventSource.close();
          setActiveEventSources((prev) => {
            const next = new Map(prev);
            next.delete(modelId);
            return next;
          });
          setDownloadProgress((prev) => {
            const next = new Map(prev);
            next.delete(modelId);
            return next;
          });
          toast.success("Model downloaded successfully", {
            description: `${data.size_mb} MB`,
          });
          // Refresh models list
          fetchModels();
          break;

        case "error":
          eventSource.close();
          setActiveEventSources((prev) => {
            const next = new Map(prev);
            next.delete(modelId);
            return next;
          });
          setDownloadProgress((prev) => {
            const next = new Map(prev);
            next.delete(modelId);
            return next;
          });
          toast.error("Download failed", {
            description: data.message || "Unknown error",
          });
          break;
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      setActiveEventSources((prev) => {
        const next = new Map(prev);
        next.delete(modelId);
        return next;
      });
      setDownloadProgress((prev) => {
        const next = new Map(prev);
        next.delete(modelId);
        return next;
      });
      toast.error("Connection error", {
        description: "Lost connection to server during download",
      });
    };
  };

  const handleCancelDownload = async (modelId: string) => {
    // Close EventSource
    const eventSource = activeEventSources.get(modelId);
    if (eventSource) {
      eventSource.close();
    }

    // Call backend to cancel
    try {
      await fetch(`/api/models/embedding/${modelId}/download`, {
        method: "DELETE",
      });

      toast.info("Download cancelled", {
        description: "Download stopped and partial files removed",
      });
    } catch (err) {
      console.error("Failed to cancel download:", err);
    }

    // Clean up state
    setActiveEventSources((prev) => {
      const next = new Map(prev);
      next.delete(modelId);
      return next;
    });
    setDownloadProgress((prev) => {
      const next = new Map(prev);
      next.delete(modelId);
      return next;
    });
  };

  const handleDelete = async (modelId: string) => {
    if (!confirm("Are you sure you want to delete this model?")) {
      return;
    }

    setDeletingModels((prev) => new Set(prev).add(modelId));

    try {
      const response = await fetch(`/api/models/embedding/${modelId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      toast.success("Model deleted", {
        description: data.message,
      });

      // Refresh models list
      await fetchModels();
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to delete model";
      toast.error("Deletion failed", {
        description: errorMsg,
      });
    } finally {
      setDeletingModels((prev) => {
        const next = new Set(prev);
        next.delete(modelId);
        return next;
      });
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Available Models</CardTitle>
          <CardDescription>Loading embedding models...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Embedding Models</CardTitle>
          <CardDescription>
            Download and manage embedding models for text vectorization
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[200px]">Model</TableHead>
                  <TableHead className="w-[120px]">Team</TableHead>
                  <TableHead className="w-[100px]">Dimensions</TableHead>
                  <TableHead className="w-[100px]">Languages</TableHead>
                  <TableHead>Description</TableHead>
                  <TableHead className="w-[100px]">Size</TableHead>
                  <TableHead className="w-[80px]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((model) => {
                  const progress = downloadProgress.get(model.id);
                  const isDownloading = !!progress;
                  const isDeleting = deletingModels.has(model.id);

                  return (
                    <TableRow key={model.id}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div className="font-medium">{model.name}</div>
                          <a
                            href={model.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-muted-foreground hover:text-foreground transition-colors"
                          >
                            <ExternalLink className="h-3.5 w-3.5" />
                          </a>
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className="text-sm">{model.team}</span>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{model.dimensions}</Badge>
                      </TableCell>
                      <TableCell>
                        <span className="text-sm">{model.languages}</span>
                      </TableCell>
                      <TableCell>
                        <span className="text-sm">{model.description}</span>
                      </TableCell>
                      <TableCell>
                        {isDownloading && progress ? (
                          <div className="space-y-1 min-w-[120px]">
                            {progress.percent !== null && progress.percent > 0 ? (
                              <>
                                <Progress value={progress.percent} className="h-2" />
                                <div className="text-xs text-muted-foreground">
                                  {progress.currentMb} MB ({progress.percent}%)
                                </div>
                              </>
                            ) : (
                              <div className="text-xs text-muted-foreground">
                                Downloading... {progress.currentMb} MB
                              </div>
                            )}
                          </div>
                        ) : model.is_downloaded ? (
                          <Badge variant="default" className="bg-green-600 hover:bg-green-700">
                            {model.downloaded_size_mb} MB
                          </Badge>
                        ) : (
                          <Badge variant="secondary">
                            {model.size_mb} MB
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        <div className="flex justify-end gap-1">
                          {isDownloading ? (
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                              onClick={() => handleCancelDownload(model.id)}
                              title="Cancel download"
                            >
                              <X className="h-4 w-4" />
                            </Button>
                          ) : !model.is_downloaded ? (
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-8 w-8 p-0"
                              onClick={() => handleDownload(model.id)}
                              title="Download model"
                            >
                              <Download className="h-4 w-4" />
                            </Button>
                          ) : (
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                              onClick={() => handleDelete(model.id)}
                              disabled={isDeleting}
                              title="Delete model"
                            >
                              {isDeleting ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Trash2 className="h-4 w-4" />
                              )}
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>About Embedding Models</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <div>
            <h4 className="font-semibold mb-2">What are embeddings?</h4>
            <p className="text-muted-foreground">
              Embeddings convert text into dense vector representations that capture semantic meaning.
              They enable powerful applications like semantic search, text similarity, and clustering.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2">Choosing a model</h4>
            <ul className="list-disc list-inside text-muted-foreground space-y-1">
              <li>
                <strong>Higher dimensions (1024)</strong>: Better quality, slower performance, more storage
              </li>
              <li>
                <strong>Lower dimensions (384)</strong>: Faster, smaller, good for resource-constrained environments
              </li>
              <li>
                <strong>Multilingual models</strong>: Support multiple languages but may sacrifice quality
              </li>
              <li>
                <strong>English-only models</strong>: Best quality for English text
              </li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
