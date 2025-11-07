"use client";

import React, { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ExternalLink } from "lucide-react";
import { toast } from "sonner";
import { getTaskDisplayName } from "@/lib/tasks";
import { LibInfo, LibsResponse } from "@/lib/libs";

export default function LibsPage() {
  const [libs, setLibs] = useState<LibInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchLibs = async () => {
    try {
      const response = await fetch("/api/libs");
      if (!response.ok) throw new Error("Failed to fetch libs");
      const data: LibsResponse = await response.json();
      setLibs(data.libs);
    } catch (error) {
      console.error("Failed to fetch libs:", error);
      toast.error("Failed to load libs");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLibs();
  }, []);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Libs</h1>
        <p className="text-muted-foreground">Built-in libraries and tools available by task</p>
      </div>

      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Lib</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead>Tasks</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={4} className="text-center py-8 text-muted-foreground">Loading libs...</TableCell>
              </TableRow>
            ) : libs.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="text-center py-8 text-muted-foreground">No libs available</TableCell>
              </TableRow>
            ) : (
              libs.map((l) => (
                <TableRow key={l.id}>
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2">
                      {l.label}
                      {l.reference_url && (
                        <a href={l.reference_url} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground">
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{l.provider || "—"}</TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {l.tasks.length > 0 ? (
                        l.tasks.map(task => (
                          <Badge key={task} variant="secondary" className="text-xs">
                            {getTaskDisplayName(task)}
                          </Badge>
                        ))
                      ) : (
                        <Badge variant="secondary" className="text-xs">Unknown</Badge>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    {l.status === "ready" ? (
                      <Badge variant="default" className="bg-green-600">Ready</Badge>
                    ) : l.status === "downloading" ? (
                      <Badge variant="default" className="bg-blue-600">Preparing</Badge>
                    ) : l.status === "failed" ? (
                      <Badge variant="destructive">Failed</Badge>
                    ) : (
                      <Badge variant="secondary">Not Downloaded</Badge>
                    )}
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

