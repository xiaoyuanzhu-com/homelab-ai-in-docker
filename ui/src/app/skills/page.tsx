"use client";

import React, { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Download, Trash2, ExternalLink, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";
import { toast } from "sonner";
import { getTaskDisplayName } from "@/lib/tasks";
import { SkillInfo, SkillsResponse } from "@/lib/skills";

interface LogEntry {
  log_line: string;
  timestamp: string;
}

export default function ModelsPage() {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloadingSkills, setDownloadingSkills] = useState<Set<string>>(new Set());
  const [expandedSkills, setExpandedSkills] = useState<Set<string>>(new Set());
  const [skillLogs, setSkillLogs] = useState<Record<string, LogEntry[]>>({});

  const fetchSkills = async () => {
    try {
      const response = await fetch("/api/skills");
      if (!response.ok) throw new Error("Failed to fetch skills");
      const data: SkillsResponse = await response.json();
      setSkills(data.skills);
    } catch (error) {
      console.error("Failed to fetch skills:", error);
      toast.error("Failed to load skills");
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchSkills();
  }, []);

  // Auto-refresh every 5s when downloads are in progress
  // Trigger when either server-reported status or local downloading state is present
  useEffect(() => {
    const hasServerDownloading = skills.some(skill => skill.status === "downloading");
    const hasLocalDownloading = downloadingSkills.size > 0;

    if (!hasServerDownloading && !hasLocalDownloading) {
      return;
    }

    const intervalId = setInterval(() => {
      fetchSkills();
    }, 5000);

    return () => clearInterval(intervalId);
  }, [skills, downloadingSkills]);

  const handleDownload = async (skillId: string) => {
    setDownloadingSkills(prev => new Set(prev).add(skillId));
    toast.info("Downloading...", {
      description: "This may take several minutes depending on size",
    });

    try {
      // Use SSE streaming endpoint for real-time progress
      const encodedId = encodeURIComponent(skillId);
      const eventSource = new EventSource(`/api/skills/download?skill=${encodedId}`);

      // Kick off a refresh immediately so the UI reflects "downloading" status promptly
      fetchSkills();

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
          toast.success("Skill is ready!");
          fetchSkills();
          setDownloadingSkills(prev => {
            const newSet = new Set(prev);
            newSet.delete(skillId);
            return newSet;
          });
        } else if (data.type === "error") {
          eventSource.close();
          toast.error("Download failed", { description: data.message });
          setDownloadingSkills(prev => {
            const newSet = new Set(prev);
            newSet.delete(skillId);
            return newSet;
          });
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        toast.error("Download connection failed");
        setDownloadingSkills(prev => {
          const newSet = new Set(prev);
          newSet.delete(skillId);
          return newSet;
        });
      };
    } catch (error) {
      console.error("Failed to download skill:", error);
      toast.error("Failed to download skill");
      setDownloadingSkills(prev => {
        const newSet = new Set(prev);
        newSet.delete(skillId);
        return newSet;
      });
    }
  };

  const handleDelete = async (skillId: string) => {
    try {
      const encodedId = encodeURIComponent(skillId);
      const response = await fetch(`/api/skills/${encodedId}`, {
        method: "DELETE",
      });

      if (!response.ok) throw new Error("Delete failed");

      toast.success("Skill deleted successfully!");
      await fetchSkills();
    } catch (error) {
      console.error("Failed to delete skill:", error);
      toast.error("Failed to delete skill");
    }
  };

  const fetchLogs = async (skillId: string) => {
    try {
      const encodedId = encodeURIComponent(skillId);
      const response = await fetch(`/api/skills/${encodedId}/logs`);
      if (!response.ok) throw new Error("Failed to fetch logs");
      const data = await response.json();
      setSkillLogs(prev => ({ ...prev, [skillId]: data.logs }));
    } catch (error) {
      console.error("Failed to fetch logs:", error);
    }
  };

  const toggleExpanded = async (skillId: string) => {
    const isExpanded = expandedSkills.has(skillId);
    const newExpanded = new Set(expandedSkills);

    if (isExpanded) {
      newExpanded.delete(skillId);
    } else {
      newExpanded.add(skillId);
      // Fetch logs when expanding
      await fetchLogs(skillId);
    }

    setExpandedSkills(newExpanded);
  };

  // Auto-refresh logs for expanded downloading models
  useEffect(() => {
    const downloadingExpanded = Array.from(expandedSkills).filter(skillId => {
      const skill = skills.find(s => s.id === skillId);
      return skill?.status === "downloading";
    });

    if (downloadingExpanded.length === 0) {
      return;
    }

    const intervalId = setInterval(() => {
      downloadingExpanded.forEach(skillId => fetchLogs(skillId));
    }, 2000); // Refresh logs every 2 seconds

    return () => clearInterval(intervalId);
  }, [expandedSkills, skills]);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Skills</h1>
        <p className="text-muted-foreground">
          Manage downloadable and built-in skills across tasks
        </p>
      </div>

      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Skill</TableHead>
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
                <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">
                  Loading skills...
                </TableCell>
              </TableRow>
            ) : skills.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">
                  No skills available
                </TableCell>
              </TableRow>
            ) : (
              skills.map((skill) => (
                <React.Fragment key={skill.id}>
                <TableRow>
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2">
                      {skill.label}
                      {skill.reference_url && (
                        <a
                          href={skill.reference_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-muted-foreground hover:text-foreground"
                        >
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{skill.provider || "—"}</TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {skill.tasks.length > 0 ? (
                        skill.tasks.map(task => (
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
                    {typeof skill.parameters_m === "number" ? `${skill.parameters_m}M` : "—"}
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {typeof skill.gpu_memory_mb === "number"
                      ? `${(skill.gpu_memory_mb / 1024).toFixed(1)} GB`
                      : "—"}
                  </TableCell>
                  <TableCell className="text-right">
                    {skill.downloaded_size_mb
                      ? `${skill.downloaded_size_mb} MB`
                      : typeof skill.size_mb === "number" ? `${skill.size_mb} MB` : "—"}
                  </TableCell>
                  <TableCell>
                    {skill.status === "init" && (
                      <Badge variant="secondary">Not Downloaded</Badge>
                    )}
                    {skill.status === "downloading" && (
                      <div className="flex items-center gap-2">
                        <Badge variant="default" className="bg-blue-600">Downloading</Badge>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleExpanded(skill.id)}
                          className="h-6 w-6 p-0"
                          title={expandedSkills.has(skill.id) ? "Hide logs" : "Show logs"}
                        >
                          {expandedSkills.has(skill.id) ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    )}
                    {skill.status === "ready" && (
                      <Badge variant="default" className="bg-green-600">Ready</Badge>
                    )}
                    {skill.status === "failed" && (
                      <Badge
                        variant="destructive"
                        title={skill.error_message || "Download failed"}
                      >
                        Failed
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell>
                    <div className="flex gap-1 justify-end">
                      {skill.status === "init" && skill.requires_download && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDownload(skill.id)}
                          disabled={downloadingSkills.has(skill.id)}
                          className="h-8 px-2"
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      )}
                      {skill.status === "downloading" && (
                        <>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDownload(skill.id)}
                            className="h-8 px-2"
                            title="Retry download"
                          >
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDelete(skill.id)}
                            className="h-8 px-2"
                            title="Delete"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </>
                      )}
                      {skill.status === "ready" && skill.requires_download && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(skill.id)}
                          className="h-8 px-2"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      )}
                      {skill.status === "failed" && (
                        <>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDownload(skill.id)}
                            className="h-8 px-2"
                            title="Retry download"
                          >
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDelete(skill.id)}
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
                {expandedSkills.has(skill.id) && (
                  <TableRow>
                    <TableCell colSpan={8} className="bg-muted/50 p-0">
                      <div className="p-4 max-h-96 overflow-y-auto">
                        <div className="text-sm font-mono bg-black text-green-400 p-4 rounded">
                          {skillLogs[skill.id] && skillLogs[skill.id].length > 0 ? (
                            skillLogs[skill.id].map((log, idx) => (
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
