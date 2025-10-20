"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Download, Trash2, CheckCircle2, Circle } from "lucide-react";
import { toast } from "sonner";

interface Model {
  id: string;
  name: string;
  team: string;
  type: string;
  license?: string;
  dimensions?: number;
  languages?: string[];
  description: string;
  size_mb: number;
  link: string;
  is_downloaded: boolean;
  downloaded_size_mb?: number;
}

interface ModelsResponse {
  models: Model[];
}

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());

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

  useEffect(() => {
    fetchModels();
  }, []);

  const handleDownload = async (modelId: string) => {
    setDownloadingModels(prev => new Set(prev).add(modelId));
    toast.info("Downloading model...", {
      description: "This may take several minutes depending on model size",
    });

    try {
      // Use streaming endpoint for embedding models
      const modelType = models.find(m => m.id === modelId)?.type;

      if (modelType === "embedding") {
        // Use SSE streaming endpoint for real-time progress
        const encodedId = encodeURIComponent(modelId);
        const eventSource = new EventSource(`/api/models/embedding/${encodedId}/download`);

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
      } else {
        // For other model types, use simple download endpoint
        const response = await fetch(`/api/models/download?model_id=${encodeURIComponent(modelId)}`, {
          method: "POST",
        });

        if (!response.ok) throw new Error("Download failed");

        toast.success("Model downloaded successfully!");
        await fetchModels();
        setDownloadingModels(prev => {
          const newSet = new Set(prev);
          newSet.delete(modelId);
          return newSet;
        });
      }
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

  const groupedModels = models.reduce((acc, model) => {
    if (!acc[model.type]) {
      acc[model.type] = [];
    }
    acc[model.type].push(model);
    return acc;
  }, {} as Record<string, Model[]>);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Model Management</h1>
        <p className="text-muted-foreground">
          Manage AI models for text embedding, image captioning, and other tasks
        </p>
      </div>

      {loading ? (
        <div className="space-y-6">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-6 w-48" />
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[1, 2].map((j) => (
                    <Skeleton key={j} className="h-24 w-full" />
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="space-y-6">
          {Object.entries(groupedModels).map(([type, typeModels]) => (
            <Card key={type}>
              <CardHeader>
                <CardTitle className="capitalize">{type} Models</CardTitle>
                <CardDescription>
                  {typeModels.filter(m => m.is_downloaded).length} of {typeModels.length} downloaded
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {typeModels.map((model) => (
                    <div
                      key={model.id}
                      className="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          {model.is_downloaded ? (
                            <CheckCircle2 className="h-5 w-5 text-green-600" />
                          ) : (
                            <Circle className="h-5 w-5 text-muted-foreground" />
                          )}
                          <h3 className="font-semibold">{model.name}</h3>
                        </div>
                        <p className="text-sm text-muted-foreground mb-2">
                          {model.description}
                        </p>
                        <div className="flex gap-2 flex-wrap">
                          <Badge variant="outline" className="text-xs">
                            {model.team}
                          </Badge>
                          {model.dimensions && (
                            <Badge variant="secondary" className="text-xs">
                              {model.dimensions}D
                            </Badge>
                          )}
                          {model.languages && model.languages.length > 0 && (
                            <Badge variant="secondary" className="text-xs">
                              {model.languages.join(", ")}
                            </Badge>
                          )}
                          <Badge variant="secondary" className="text-xs">
                            {model.is_downloaded && model.downloaded_size_mb
                              ? `${model.downloaded_size_mb} MB`
                              : `~${model.size_mb} MB`}
                          </Badge>
                          {model.is_downloaded && (
                            <Badge variant="default" className="text-xs">
                              Downloaded
                            </Badge>
                          )}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {model.is_downloaded ? (
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => handleDelete(model.id)}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </Button>
                        ) : (
                          <Button
                            variant="default"
                            size="sm"
                            onClick={() => handleDownload(model.id)}
                            disabled={downloadingModels.has(model.id)}
                          >
                            <Download className="h-4 w-4 mr-2" />
                            {downloadingModels.has(model.id) ? "Downloading..." : "Download"}
                          </Button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
