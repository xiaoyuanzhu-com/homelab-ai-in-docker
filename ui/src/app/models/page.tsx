"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Download, Trash2, ExternalLink } from "lucide-react";
import { toast } from "sonner";

interface Model {
  id: string;
  name: string;
  team: string;
  type: string;
  task: string;
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
              <TableHead className="text-right">Size</TableHead>
              <TableHead className="w-[100px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                  Loading models...
                </TableCell>
              </TableRow>
            ) : models.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                  No models available
                </TableCell>
              </TableRow>
            ) : (
              models.map((model) => (
                <TableRow key={model.id}>
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
                      {model.task}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {model.is_downloaded && model.downloaded_size_mb
                      ? `${model.downloaded_size_mb} MB`
                      : `${model.size_mb} MB`}
                  </TableCell>
                  <TableCell>
                    <div className="flex gap-1 justify-end">
                      {model.is_downloaded ? (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(model.id)}
                          className="h-8 px-2"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      ) : (
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
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
