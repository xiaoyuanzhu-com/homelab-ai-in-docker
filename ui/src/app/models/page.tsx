"use client";

import React, { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Download, Trash2, ExternalLink, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";
import { toast } from "sonner";
import { getTaskDisplayName } from "@/lib/tasks";
import { ModelInfo, ModelsResponse } from "@/lib/models";

interface LogEntry {
  log_line: string;
  timestamp: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [logs, setLogs] = useState<Record<string, LogEntry[]>>({});
  const [toDelete, setToDelete] = useState<ModelInfo | null>(null);

  const fetchModels = async () => {
    try {
      const response = await fetch("/api/models");
      if (!response.ok) {
        // Try to extract JSON error; fallback to text
        try {
          const err = await response.json();
          throw new Error(err?.detail || "Failed to fetch models");
        } catch {
          const text = await response.text();
          throw new Error(text || `HTTP ${response.status}`);
        }
      }
      const contentType = response.headers.get("content-type") || "";
      if (!contentType.includes("application/json")) {
        const text = await response.text();
        throw new Error(`Unexpected response (not JSON): ${text.slice(0, 200)}`);
      }
      const data: ModelsResponse = await response.json();
      setModels(data.models);
    } catch (error) {
      console.error("Failed to fetch models:", error);
      toast.error("Failed to load models", { description: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  useEffect(() => {
    const hasServerDownloading = models.some(m => m.status === "downloading");
    const hasLocalDownloading = downloading.size > 0;
    if (!hasServerDownloading && !hasLocalDownloading) return;
    const id = setInterval(() => fetchModels(), 5000);
    return () => clearInterval(id);
  }, [models, downloading]);

  const handleDownload = async (modelId: string) => {
    setDownloading(prev => new Set(prev).add(modelId));
    toast.info("Downloading...", { description: "This may take several minutes depending on size" });
    try {
      const encoded = encodeURIComponent(modelId);
      const es = new EventSource(`/api/models/download?model=${encoded}`);
      fetchModels();
      es.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "progress") {
          const message = data.current_mb ? `Downloaded ${data.current_mb} MB...` : "Downloading...";
          toast.info(message);
        } else if (data.type === "complete") {
          es.close();
          toast.success("Model is ready!");
          fetchModels();
          setDownloading(prev => { const s = new Set(prev); s.delete(modelId); return s; });
        } else if (data.type === "error") {
          es.close();
          toast.error("Download failed", { description: data.message });
          setDownloading(prev => { const s = new Set(prev); s.delete(modelId); return s; });
        }
      };
      es.onerror = () => {
        es.close();
        toast.error("Download connection failed");
        setDownloading(prev => { const s = new Set(prev); s.delete(modelId); return s; });
      };
    } catch (error) {
      console.error("Failed to download model:", error);
      toast.error("Failed to download model");
      setDownloading(prev => { const s = new Set(prev); s.delete(modelId); return s; });
    }
  };

  const confirmDelete = async () => {
    if (!toDelete) return;
    try {
      const encoded = encodeURIComponent(toDelete.id);
      const response = await fetch(`/api/models/${encoded}`, { method: "DELETE" });
      if (!response.ok) throw new Error("Delete failed");
      toast.success("Model deleted successfully!");
      await fetchModels();
    } catch (error) {
      console.error("Failed to delete model:", error);
      toast.error("Failed to delete model");
    } finally {
      setToDelete(null);
    }
  };

  const fetchLogs = async (modelId: string) => {
    try {
      const encoded = encodeURIComponent(modelId);
      const response = await fetch(`/api/models/${encoded}/logs`);
      if (!response.ok) throw new Error("Failed to fetch logs");
      const data = await response.json();
      setLogs(prev => ({ ...prev, [modelId]: data.logs }));
    } catch (error) {
      console.error("Failed to fetch logs:", error);
    }
  };

  const toggleExpanded = async (modelId: string) => {
    const isExpanded = expanded.has(modelId);
    const next = new Set(expanded);
    if (isExpanded) next.delete(modelId); else { next.add(modelId); await fetchLogs(modelId); }
    setExpanded(next);
  };

  useEffect(() => {
    const downloadingExpanded = Array.from(expanded).filter(id => models.find(s => s.id === id)?.status === "downloading");
    if (downloadingExpanded.length === 0) return;
    const id = setInterval(() => { downloadingExpanded.forEach(m => fetchLogs(m)); }, 2000);
    return () => clearInterval(id);
  }, [expanded, models]);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Models</h1>
        <p className="text-muted-foreground">Manage downloadable AI models across tasks</p>
      </div>

      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead>Tasks</TableHead>
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
                <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">Loading models...</TableCell>
              </TableRow>
            ) : models.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">No models available</TableCell>
              </TableRow>
            ) : (
              [...models].sort((a, b) => {
                const taskA = a.tasks[0] ? getTaskDisplayName(a.tasks[0]) : "zzz";
                const taskB = b.tasks[0] ? getTaskDisplayName(b.tasks[0]) : "zzz";
                if (taskA !== taskB) return taskA.localeCompare(taskB);
                return a.label.localeCompare(b.label);
              }).map((m) => (
                <React.Fragment key={m.id}>
                  <TableRow>
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        {m.label}
                        {m.reference_url && (
                          <a href={m.reference_url} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground">
                            <ExternalLink className="h-3 w-3" />
                          </a>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-muted-foreground">{m.provider || "—"}</TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {m.tasks.length > 0 ? (
                          m.tasks.map(task => (
                            <Badge key={task} variant="secondary" className="text-xs">
                              {getTaskDisplayName(task)}
                            </Badge>
                          ))
                        ) : (
                          <Badge variant="secondary" className="text-xs">Unknown</Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-right text-muted-foreground">
                      {typeof m.parameters_m === "number" ? `${m.parameters_m}M` : "—"}
                    </TableCell>
                    <TableCell className="text-right text-muted-foreground">
                      {typeof m.gpu_memory_mb === "number" ? `${(m.gpu_memory_mb / 1024).toFixed(1)} GB` : "—"}
                    </TableCell>
                    <TableCell className="text-right">
                      {m.downloaded_size_mb ? `${m.downloaded_size_mb} MB` : typeof m.size_mb === "number" ? `${m.size_mb} MB` : "—"}
                    </TableCell>
                    <TableCell>
                      {m.status === "init" && (<Badge variant="secondary">Not Downloaded</Badge>)}
                      {m.status === "downloading" && (
                        <div className="flex items-center gap-2">
                          <Badge variant="default" className="bg-blue-600">Downloading</Badge>
                          <Button variant="ghost" size="sm" onClick={() => toggleExpanded(m.id)} className="h-6 w-6 p-0" title={expanded.has(m.id) ? "Hide logs" : "Show logs"}>
                            {expanded.has(m.id) ? (<ChevronUp className="h-4 w-4" />) : (<ChevronDown className="h-4 w-4" />)}
                          </Button>
                        </div>
                      )}
                      {m.status === "ready" && (<Badge variant="default" className="bg-green-600">Ready</Badge>)}
                      {m.status === "failed" && (<Badge variant="destructive" title={m.error_message || "Download failed"}>Failed</Badge>)}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1 justify-end">
                        {m.status === "init" && m.requires_download && (
                          <Button variant="ghost" size="sm" onClick={() => handleDownload(m.id)} disabled={downloading.has(m.id)} className="h-8 px-2">
                            <Download className="h-4 w-4" />
                          </Button>
                        )}
                        {m.status === "ready" && (
                          <Button variant="ghost" size="sm" onClick={() => fetchModels()} className="h-8 px-2" title="Refresh">
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                        )}
                        {m.status !== "downloading" && (
                          <Button variant="ghost" size="sm" onClick={() => setToDelete(m)} className="h-8 px-2" title="Delete assets">
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>

                  {expanded.has(m.id) && (
                    <TableRow>
                      <TableCell colSpan={8} className="bg-muted/30 p-0">
                        <div className="p-3">
                          <div className="text-xs text-muted-foreground mb-2">Download Logs</div>
                          <pre className="text-xs whitespace-pre-wrap leading-5">
                            {logs[m.id]?.map((l) => `[${l.timestamp}] ${l.log_line}`).join("\n") || "No logs yet..."}
                          </pre>
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

      <AlertDialog open={!!toDelete} onOpenChange={(open) => !open && setToDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete model assets?</AlertDialogTitle>
            <AlertDialogDescription>
              This will remove any downloaded files for the model from disk. You can download it again later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDelete}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
