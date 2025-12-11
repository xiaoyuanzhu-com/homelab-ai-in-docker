"use client";

import React, { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Download, Trash2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { EnvInfo, EnvsResponse } from "@/lib/envs";

export default function EnvsPage() {
  const [envs, setEnvs] = useState<Record<string, EnvInfo>>({});
  const [loading, setLoading] = useState(true);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);

  const fetchEnvs = async () => {
    try {
      const response = await fetch("/api/environments");
      if (!response.ok) {
        try {
          const err = await response.json();
          throw new Error(err?.detail || "Failed to fetch environments");
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
      const data: EnvsResponse = await response.json();
      setEnvs(data.environments);
    } catch (error) {
      console.error("Failed to fetch environments:", error);
      toast.error("Failed to load environments", { description: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  };

  const installEnv = async (envId: string) => {
    setActionInProgress(envId);
    try {
      const response = await fetch(`/api/environments/${envId}/install`, {
        method: "POST",
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err?.detail?.message || "Failed to install environment");
      }
      toast.success(`Installing ${envId}`, { description: "Installation started in background" });
      await fetchEnvs();
    } catch (error) {
      toast.error(`Failed to install ${envId}`, { description: error instanceof Error ? error.message : String(error) });
    } finally {
      setActionInProgress(null);
    }
  };

  const deleteEnv = async (envId: string) => {
    setActionInProgress(envId);
    try {
      const response = await fetch(`/api/environments/${envId}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err?.detail?.message || "Failed to delete environment");
      }
      const result = await response.json();
      toast.success(`Deleted ${envId}`, { description: result.freed_mb ? `Freed ${result.freed_mb.toFixed(1)} MB` : "Environment removed" });
      await fetchEnvs();
    } catch (error) {
      toast.error(`Failed to delete ${envId}`, { description: error instanceof Error ? error.message : String(error) });
    } finally {
      setActionInProgress(null);
    }
  };

  useEffect(() => {
    fetchEnvs();
    // Poll for updates every 5 seconds when there are installing envs
    const interval = setInterval(() => {
      const hasInstalling = Object.values(envs).some(e => e.status === "installing");
      if (hasInstalling) {
        fetchEnvs();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [envs]);

  const envList = Object.entries(envs).sort(([a], [b]) => a.localeCompare(b));

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Environments</h1>
        <p className="text-muted-foreground">Worker environments for running ML tasks</p>
      </div>

      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Environment</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Python</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">Loading environments...</TableCell>
              </TableRow>
            ) : envList.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">No environments available</TableCell>
              </TableRow>
            ) : (
              envList.map(([envId, env]) => (
                <TableRow key={envId}>
                  <TableCell className="font-medium">{envId}</TableCell>
                  <TableCell>
                    {env.status === "ready" ? (
                      <Badge variant="default" className="bg-green-600">Ready</Badge>
                    ) : env.status === "installing" ? (
                      <Badge variant="default" className="bg-blue-600">
                        <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                        Installing
                      </Badge>
                    ) : env.status === "failed" ? (
                      <Badge variant="destructive" title={env.error_message || undefined}>Failed</Badge>
                    ) : (
                      <Badge variant="secondary">Not Installed</Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {env.size_mb ? `${env.size_mb.toFixed(1)} MB` : "—"}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {env.python_version || "—"}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      {(env.status === "not_installed" || env.status === "failed") && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => installEnv(envId)}
                          disabled={actionInProgress === envId}
                        >
                          {actionInProgress === envId ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Download className="h-4 w-4" />
                          )}
                        </Button>
                      )}
                      {env.status === "ready" && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => deleteEnv(envId)}
                          disabled={actionInProgress === envId}
                        >
                          {actionInProgress === envId ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
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
