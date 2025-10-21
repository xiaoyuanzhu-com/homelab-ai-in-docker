"use client";

import React, { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Download, Trash2, ExternalLink, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";
import { toast } from "sonner";
import { getTaskDisplayName } from "@/lib/tasks";

interface Model {
  id: string;
  name: string;
  team: string;
  type: string;
  task: string;
  size_mb: number;
  parameters_m: number;
  gpu_memory_mb: number;
  link: string;
  status: "init" | "downloading" | "failed" | "downloaded";
  downloaded_size_mb?: number;
  error_message?: string;
}

interface ModelsResponse {
  models: Model[];
}

interface LogEntry {
  log_line: string;
  timestamp: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());
  const [modelLogs, setModelLogs] = useState<Record<string, LogEntry[]>>({});

  const fetchModels = async () => {
    try {
      const response = await fetch("/api/models");
      if (!response.ok) throw new Error("Failed to fetch models");
      const data: ModelsResponse = await response.json();
      setModels(data.models);
    } catch (error) {
      console.error("Failed to fetch models:", error);
      toast.error("Failed to load models");
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchModels();
  }, []);

  // Auto-refresh every 5s when there are downloading models
  useEffect(() => {
    const hasDownloading = models.some(m => m.status === "downloading");

    if (!hasDownloading) {
      return;
    }

    const intervalId = setInterval(() => {
      fetchModels();
    }, 5000);

    return () => clearInterval(intervalId);
  }, [models]);

  const handleDownload = async (modelId: string) => {
    setDownloadingModels(prev => new Set(prev).add(modelId));
    toast.info("Downloading model...", {
      description: "This may take several minutes depending on model size",
    });

    try {
      // Use SSE streaming endpoint for real-time progress
      const encodedId = encodeURIComponent(modelId);
      const eventSource = new EventSource(`/api/models/download?model=${encodedId}`);

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "progress") {
          // Update progress in toast
          const message = data.current_mb
            ? `Downloaded ${data.current_mb} MB...`
            : "Downloading...";
          toast.info(message);
        } else if (data.type === "complete") {
          eventSource.close();
          toast.success("Model downloaded successfully!");
          fetchModels();
          setDownloadingModels(prev => {
            const newSet = new Set(prev);
            newSet.delete(modelId);
            return newSet;
          });
        } else if (data.type === "error") {
          eventSource.close();
          toast.error("Download failed", { description: data.message });
          setDownloadingModels(prev => {
            const newSet = new Set(prev);
            newSet.delete(modelId);
            return newSet;
          });
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        toast.error("Download connection failed");
        setDownloadingModels(prev => {
          const newSet = new Set(prev);
          newSet.delete(modelId);
          return newSet;
        });
      };
    } catch (error) {
      console.error("Failed to download model:", error);
      toast.error("Failed to download model");
      setDownloadingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(modelId);
        return newSet;
      });
    }
  };

  const handleDelete = async (modelId: string) => {
    try {
      const encodedId = encodeURIComponent(modelId);
      const response = await fetch(`/api/models/${encodedId}`, {
        method: "DELETE",
      });

      if (!response.ok) throw new Error("Delete failed");

      toast.success("Model deleted successfully!");
      await fetchModels();
    } catch (error) {
      console.error("Failed to delete model:", error);
      toast.error("Failed to delete model");
    }
  };

  const fetchLogs = async (modelId: string) => {
    try {
      const encodedId = encodeURIComponent(modelId);
      const response = await fetch(`/api/models/${encodedId}/logs`);
      if (!response.ok) throw new Error("Failed to fetch logs");
      const data = await response.json();
      setModelLogs(prev => ({ ...prev, [modelId]: data.logs }));
    } catch (error) {
      console.error("Failed to fetch logs:", error);
    }
  };

  const toggleExpanded = async (modelId: string) => {
    const isExpanded = expandedModels.has(modelId);
    const newExpanded = new Set(expandedModels);

    if (isExpanded) {
      newExpanded.delete(modelId);
    } else {
      newExpanded.add(modelId);
      // Fetch logs when expanding
      await fetchLogs(modelId);
    }

    setExpandedModels(newExpanded);
  };

  // Auto-refresh logs for expanded downloading models
  useEffect(() => {
    const downloadingExpanded = Array.from(expandedModels).filter(modelId => {
      const model = models.find(m => m.id === modelId);
      return model?.status === "downloading";
    });

    if (downloadingExpanded.length === 0) {
      return;
    }

    const intervalId = setInterval(() => {
      downloadingExpanded.forEach(modelId => fetchLogs(modelId));
    }, 2000); // Refresh logs every 2 seconds

    return () => clearInterval(intervalId);
  }, [expandedModels, models]);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Models</h1>
        <p className="text-muted-foreground">
          Manage AI models for various tasks
        </p>
      </div>

      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Team</TableHead>
              <TableHead>Task</TableHead>
              <TableHead className="text-right">Parameters</TableHead>
              <TableHead className="text-right">GPU Memory</TableHead>
              <TableHead className="text-right">Size</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="w-[120px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">
                  Loading models...
                </TableCell>
              </TableRow>
            ) : models.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">
                  No models available
                </TableCell>
              </TableRow>
            ) : (
              models.map((model) => (
                <React.Fragment key={model.id}>
                <TableRow>
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2">
                      {model.name}
                      <a
                        href={model.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-muted-foreground hover:text-foreground"
                      >
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{model.team}</TableCell>
                  <TableCell>
                    <Badge variant="secondary" className="text-xs">
                      {getTaskDisplayName(model.task)}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {model.parameters_m}M
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {(model.gpu_memory_mb / 1024).toFixed(1)} GB
                  </TableCell>
                  <TableCell className="text-right">
                    {model.downloaded_size_mb
                      ? `${model.downloaded_size_mb} MB`
                      : `${model.size_mb} MB`}
                  </TableCell>
                  <TableCell>
                    {model.status === "init" && (
                      <Badge variant="secondary">Not Downloaded</Badge>
                    )}
                    {model.status === "downloading" && (
                      <div className="flex items-center gap-2">
                        <Badge variant="default" className="bg-blue-600">Downloading</Badge>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleExpanded(model.id)}
                          className="h-6 w-6 p-0"
                          title={expandedModels.has(model.id) ? "Hide logs" : "Show logs"}
                        >
                          {expandedModels.has(model.id) ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    )}
                    {model.status === "downloaded" && (
                      <Badge variant="default" className="bg-green-600">Downloaded</Badge>
                    )}
                    {model.status === "failed" && (
                      <Badge
                        variant="destructive"
                        title={model.error_message || "Download failed"}
                      >
                        Failed
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell>
                    <div className="flex gap-1 justify-end">
                      {model.status === "init" && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDownload(model.id)}
                          disabled={downloadingModels.has(model.id)}
                          className="h-8 px-2"
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      )}
                      {model.status === "downloading" && (
                        <>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDownload(model.id)}
                            disabled={downloadingModels.has(model.id)}
                            className="h-8 px-2"
                            title="Retry download"
                          >
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDelete(model.id)}
                            className="h-8 px-2"
                            title="Delete"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </>
                      )}
                      {model.status === "downloaded" && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(model.id)}
                          className="h-8 px-2"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      )}
                      {model.status === "failed" && (
                        <>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDownload(model.id)}
                            disabled={downloadingModels.has(model.id)}
                            className="h-8 px-2"
                            title="Retry download"
                          >
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDelete(model.id)}
                            className="h-8 px-2"
                            title="Delete"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
                {/* Expandable logs row */}
                {expandedModels.has(model.id) && (
                  <TableRow>
                    <TableCell colSpan={8} className="bg-muted/50 p-0">
                      <div className="p-4 max-h-96 overflow-y-auto">
                        <div className="text-sm font-mono bg-black text-green-400 p-4 rounded">
                          {modelLogs[model.id] && modelLogs[model.id].length > 0 ? (
                            modelLogs[model.id].map((log, idx) => (
                              <div key={idx} className="whitespace-pre-wrap break-all">
                                {log.log_line}
                              </div>
                            ))
                          ) : (
                            <div className="text-muted-foreground">No logs available yet...</div>
                          )}
                        </div>
                      </div>
                    </TableCell>
                  </TableRow>
                )}
                </React.Fragment>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
