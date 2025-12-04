"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { TaskHistoryList } from "@/components/task-history";
import { ChevronDown, ChevronUp } from "lucide-react";

interface HardwareStats {
  cpu: {
    usage_percent: number;
    cores: number;
    frequency_mhz: number | null;
    model: string | null;
    temperature_c: number | null;
  };
  memory: {
    total_gb: number;
    available_gb: number;
    used_gb: number;
    usage_percent: number;
  };
  gpu: {
    available: boolean;
    count: number;
    devices: Array<{
      id: number;
      name: string;
      driver_version?: string | null;
      cuda_version?: string | null;
      total_memory_gb?: number;
      used_memory_gb?: number;
      free_memory_gb?: number;
      memory_usage_percent?: number;
      utilization_percent?: number | null;
      memory_utilization_percent?: number | null;
      temperature_c?: number | null;
      pytorch_allocated_gb?: number | null;
      pytorch_reserved_gb?: number | null;
    }>;
  };
  inference: {
    device: string;
    description: string;
  };
}

interface TaskStats {
  running: number;
  today: number;
  total: number;
}

interface GPUMemoryDetails {
  devices: Array<{
    device_id: number;
    name: string;
    nvml_stats: {
      total_mb: number;
      used_mb: number;
      free_mb: number;
      usage_percent: number;
    };
    pytorch_stats: {
      allocated_mb: number;
      reserved_mb: number;
      cached_mb: number;
      explanation: {
        allocated: string;
        reserved: string;
        cached: string;
      };
    };
    memory_summary: string | null;
  }>;
  loaded_models: Array<{
    model_id: string;
    size_mb: number;
  }>;
  tip: string;
}

// Helper function to get color based on usage percentage
function getUsageColor(usage: number): string {
  if (usage < 25) return "bg-green-600";
  if (usage < 50) return "bg-yellow-600";
  if (usage < 75) return "bg-orange-600";
  return "bg-red-600";
}

export default function StatsPage() {
  const [hardwareStats, setHardwareStats] = useState<HardwareStats | null>(null);
  const [taskStats, setTaskStats] = useState<TaskStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [gpuMemoryExpanded, setGpuMemoryExpanded] = useState(false);
  const [gpuMemoryDetails, setGpuMemoryDetails] = useState<GPUMemoryDetails | null>(null);
  const [loadingGpuDetails, setLoadingGpuDetails] = useState(false);

  useEffect(() => {
    let isMounted = true;
    let timeoutId: NodeJS.Timeout | null = null;

    const fetchStats = async () => {
      try {
        // Fetch hardware stats
        const hardwareRes = await fetch("/api/hardware");
        const hardwareData = await hardwareRes.json();
        if (!hardwareData.error) {
          setHardwareStats(hardwareData);
        }

        // Fetch task stats
        const taskRes = await fetch("/api/history/stats");
        const taskData = await taskRes.json();
        setTaskStats(taskData);
      } catch (err) {
        console.error("Failed to fetch stats:", err);
      } finally {
        setLoading(false);
        // Schedule next refresh only if component is still mounted
        if (isMounted) {
          timeoutId = setTimeout(fetchStats, 1000);
        }
      }
    };

    // Start initial fetch
    fetchStats();

    // Cleanup: clear timeout and prevent future refreshes on unmount
    return () => {
      isMounted = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, []);

  const handleGpuMemoryInspect = async () => {
    if (gpuMemoryExpanded) {
      setGpuMemoryExpanded(false);
      return;
    }

    setGpuMemoryExpanded(true);
    setLoadingGpuDetails(true);

    try {
      const res = await fetch("/api/hardware/gpu/memory");
      const data = await res.json();
      if (!data.error) {
        setGpuMemoryDetails(data);
      }
    } catch (err) {
      console.error("Failed to fetch GPU memory details:", err);
    } finally {
      setLoadingGpuDetails(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">System Status</h1>
        <p className="text-muted-foreground">
          Real-time monitoring of hardware resources and task processing
        </p>
      </div>

      {/* Hardware Stats - Compact 2x3 Grid */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Hardware Resources</CardTitle>
          <CardDescription>
            {hardwareStats?.inference.description || "Real-time system monitoring"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="grid md:grid-cols-2 gap-4">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : hardwareStats ? (
            <div className="grid md:grid-cols-2 gap-4">
              {/* Row 1: CPU Info & GPU Info */}
              <div className="p-4 bg-muted/30 rounded-lg">
                <h3 className="text-sm font-medium mb-2">CPU</h3>
                {hardwareStats.cpu.model ? (
                  <>
                    <p className="text-lg font-bold">{hardwareStats.cpu.model}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {hardwareStats.cpu.cores} cores
                      {hardwareStats.cpu.frequency_mhz && ` @ ${hardwareStats.cpu.frequency_mhz} MHz`}
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-2xl font-bold">{hardwareStats.cpu.cores} Cores</p>
                    {hardwareStats.cpu.frequency_mhz && (
                      <p className="text-xs text-muted-foreground mt-1">
                        {hardwareStats.cpu.frequency_mhz} MHz
                      </p>
                    )}
                  </>
                )}
              </div>

              <div className="p-4 bg-muted/30 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium">GPU</h3>
                  <Badge variant={hardwareStats.gpu.available ? "default" : "secondary"} className="text-xs">
                    {hardwareStats.gpu.available ? "Available" : "Not Available"}
                  </Badge>
                </div>
                {hardwareStats.gpu.available && hardwareStats.gpu.devices.length > 0 ? (
                  <>
                    <p className="text-lg font-bold">{hardwareStats.gpu.devices[0].name}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {hardwareStats.gpu.devices[0].driver_version && hardwareStats.gpu.devices[0].cuda_version
                        ? `Driver ${hardwareStats.gpu.devices[0].driver_version} • CUDA ${hardwareStats.gpu.devices[0].cuda_version}`
                        : hardwareStats.gpu.devices[0].driver_version
                        ? `Driver ${hardwareStats.gpu.devices[0].driver_version}`
                        : hardwareStats.gpu.devices[0].cuda_version
                        ? `CUDA ${hardwareStats.gpu.devices[0].cuda_version}`
                        : "Version info unavailable"}
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">Not detected</p>
                )}
              </div>

              {/* Row 2: CPU Usage & GPU Usage */}
              <div className="p-4 bg-muted/30 rounded-lg">
                <h3 className="text-sm font-medium mb-2">CPU Usage</h3>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-2xl font-bold">{hardwareStats.cpu.usage_percent.toFixed(1)}%</span>
                  {hardwareStats.cpu.temperature_c !== undefined && hardwareStats.cpu.temperature_c !== null && (
                    <span className="text-xs text-muted-foreground">
                      {hardwareStats.cpu.temperature_c.toFixed(1)}°C
                    </span>
                  )}
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                  <div
                    className={`${getUsageColor(hardwareStats.cpu.usage_percent)} h-2 rounded-full transition-all duration-300`}
                    style={{ width: `${hardwareStats.cpu.usage_percent}%` }}
                  ></div>
                </div>
              </div>

              <div className="p-4 bg-muted/30 rounded-lg">
                <h3 className="text-sm font-medium mb-2">GPU Usage</h3>
                {hardwareStats.gpu.available && hardwareStats.gpu.devices.length > 0 ? (
                  <>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-2xl font-bold">
                        {hardwareStats.gpu.devices[0].utilization_percent ?? 0}%
                      </span>
                      {hardwareStats.gpu.devices[0].temperature_c !== undefined &&
                       hardwareStats.gpu.devices[0].temperature_c !== null && (
                        <span className="text-xs text-muted-foreground">
                          {hardwareStats.gpu.devices[0].temperature_c}°C
                        </span>
                      )}
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div
                        className={`${getUsageColor(hardwareStats.gpu.devices[0].utilization_percent ?? 0)} h-2 rounded-full transition-all duration-300`}
                        style={{ width: `${hardwareStats.gpu.devices[0].utilization_percent ?? 0}%` }}
                      ></div>
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">No GPU available</p>
                )}
              </div>

              {/* Row 3: System Memory & GPU Memory */}
              <div className="p-4 bg-muted/30 rounded-lg">
                <h3 className="text-sm font-medium mb-2">System Memory</h3>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-2xl font-bold">{hardwareStats.memory.usage_percent.toFixed(1)}%</span>
                  <span className="text-xs text-muted-foreground">
                    {hardwareStats.memory.used_gb.toFixed(1)} / {hardwareStats.memory.total_gb.toFixed(1)} GB
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                  <div
                    className={`${getUsageColor(hardwareStats.memory.usage_percent)} h-2 rounded-full transition-all duration-300`}
                    style={{ width: `${hardwareStats.memory.usage_percent}%` }}
                  ></div>
                </div>
              </div>

              <div className="p-4 bg-muted/30 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium">GPU Memory</h3>
                  {hardwareStats.gpu.available && hardwareStats.gpu.devices.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleGpuMemoryInspect}
                      className="h-6 px-2 text-xs"
                    >
                      {gpuMemoryExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                      <span className="ml-1">Inspect</span>
                    </Button>
                  )}
                </div>
                {hardwareStats.gpu.available && hardwareStats.gpu.devices.length > 0 ? (
                  <>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-2xl font-bold">
                        {hardwareStats.gpu.devices[0].memory_usage_percent?.toFixed(1) ?? '0.0'}%
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {hardwareStats.gpu.devices[0].used_memory_gb?.toFixed(1) ?? '0.0'} / {hardwareStats.gpu.devices[0].total_memory_gb?.toFixed(1) ?? '0.0'} GB
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div
                        className={`${getUsageColor(hardwareStats.gpu.devices[0].memory_usage_percent ?? 0)} h-2 rounded-full transition-all duration-300`}
                        style={{ width: `${hardwareStats.gpu.devices[0].memory_usage_percent ?? 0}%` }}
                      ></div>
                    </div>
                    {hardwareStats.gpu.devices[0].pytorch_reserved_gb !== undefined &&
                     hardwareStats.gpu.devices[0].pytorch_reserved_gb !== null &&
                     hardwareStats.gpu.devices[0].pytorch_reserved_gb > 0 && (
                      <p className="text-xs text-muted-foreground mt-1">
                        PyTorch: {hardwareStats.gpu.devices[0].pytorch_reserved_gb.toFixed(2)} GB cached
                      </p>
                    )}

                    {/* Expanded GPU Memory Details */}
                    {gpuMemoryExpanded && (
                      <div className="mt-4 pt-4 border-t border-muted">
                        {loadingGpuDetails ? (
                          <Skeleton className="h-32 w-full" />
                        ) : gpuMemoryDetails ? (
                          <div className="space-y-3">
                            {gpuMemoryDetails.devices.map((device) => (
                              <div key={device.device_id} className="space-y-2">
                                <div className="grid grid-cols-2 gap-2 text-xs">
                                  <div>
                                    <span className="text-muted-foreground">NVML Total:</span>
                                    <span className="ml-1 font-medium">{device.nvml_stats.total_mb.toFixed(0)} MB</span>
                                  </div>
                                  <div>
                                    <span className="text-muted-foreground">NVML Used:</span>
                                    <span className="ml-1 font-medium">{device.nvml_stats.used_mb.toFixed(0)} MB</span>
                                  </div>
                                  <div>
                                    <span className="text-muted-foreground">PyTorch Allocated:</span>
                                    <span className="ml-1 font-medium">{(device.pytorch_stats.allocated_mb ?? 0).toFixed(0)} MB</span>
                                  </div>
                                  <div>
                                    <span className="text-muted-foreground">PyTorch Reserved:</span>
                                    <span className="ml-1 font-medium">{(device.pytorch_stats.reserved_mb ?? 0).toFixed(0)} MB</span>
                                  </div>
                                  <div className="col-span-2">
                                    <span className="text-muted-foreground">PyTorch Cached:</span>
                                    <span className="ml-1 font-medium">{(device.pytorch_stats.cached_mb ?? 0).toFixed(0)} MB</span>
                                    <span className="ml-1 text-muted-foreground text-xs">
                                      (can be freed)
                                    </span>
                                  </div>
                                </div>

                                {gpuMemoryDetails.loaded_models.length > 0 && (
                                  <div className="mt-3">
                                    <p className="text-xs font-medium mb-1">Loaded Models:</p>
                                    <div className="space-y-1">
                                      {gpuMemoryDetails.loaded_models.map((model, idx) => (
                                        <div key={idx} className="text-xs text-muted-foreground flex justify-between">
                                          <span>{model.model_id}</span>
                                          <span>{model.size_mb.toFixed(0)} MB</span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                <p className="text-xs text-muted-foreground italic mt-2">
                                  {gpuMemoryDetails.tip}
                                </p>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-xs text-muted-foreground">Failed to load details</p>
                        )}
                      </div>
                    )}
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">No GPU available</p>
                )}
              </div>
            </div>
          ) : (
            <p className="text-center text-muted-foreground">No hardware statistics available</p>
          )}
        </CardContent>
      </Card>

      {/* Task Stats Card */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Task Statistics</CardTitle>
          <CardDescription>AI processing tasks overview</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="grid grid-cols-3 gap-8">
              {[1, 2, 3].map((i) => (
                <div key={i} className="text-center">
                  <Skeleton className="h-12 w-24 mx-auto mb-2" />
                  <Skeleton className="h-4 w-16 mx-auto" />
                </div>
              ))}
            </div>
          ) : taskStats ? (
            <div className="grid grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-4xl font-bold">{taskStats.running}</div>
                <div className="text-sm text-muted-foreground mt-1">Running</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold">{taskStats.today}</div>
                <div className="text-sm text-muted-foreground mt-1">Today</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold">{taskStats.total}</div>
                <div className="text-sm text-muted-foreground mt-1">Total</div>
              </div>
            </div>
          ) : (
            <p className="text-center text-muted-foreground">No task statistics available</p>
          )}
        </CardContent>
      </Card>

      {/* Task History */}
      <div className="mb-8">
        <TaskHistoryList />
      </div>
    </div>
  );
}
