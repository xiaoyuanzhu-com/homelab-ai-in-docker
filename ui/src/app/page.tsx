"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface HardwareStats {
  cpu: {
    usage_percent: number;
    cores: number;
    frequency_mhz: number | null;
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
      compute_capability: string;
      total_memory_gb: number;
      allocated_memory_gb: number;
      reserved_memory_gb: number;
      free_memory_gb: number;
      memory_usage_percent: number;
      utilization_percent?: number;
      temperature_c?: number;
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

export default function Home() {
  const [hardwareStats, setHardwareStats] = useState<HardwareStats | null>(null);
  const [taskStats, setTaskStats] = useState<TaskStats | null>(null);

  useEffect(() => {
    // Fetch hardware stats
    fetch("/api/hardware")
      .then((res) => res.json())
      .then((data) => {
        if (!data.error) {
          setHardwareStats(data);
        }
      })
      .catch((err) => {
        console.error("Failed to fetch hardware stats:", err);
      });

    // Fetch task stats
    fetch("/api/history/stats")
      .then((res) => res.json())
      .then((data) => {
        setTaskStats(data);
      })
      .catch((err) => {
        console.error("Failed to fetch task stats:", err);
      });

    // Refresh stats every 5 seconds
    const interval = setInterval(() => {
      fetch("/api/hardware")
        .then((res) => res.json())
        .then((data) => {
          if (!data.error) {
            setHardwareStats(data);
          }
        })
        .catch((err) => {
          console.error("Failed to fetch hardware stats:", err);
        });

      fetch("/api/history/stats")
        .then((res) => res.json())
        .then((data) => {
          setTaskStats(data);
        })
        .catch((err) => {
          console.error("Failed to fetch task stats:", err);
        });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Task Stats Card */}
      {taskStats && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Tasks</CardTitle>
            <CardDescription>AI processing tasks overview</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">{taskStats.running}</div>
                <div className="text-sm text-muted-foreground mt-1">Running</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">{taskStats.today}</div>
                <div className="text-sm text-muted-foreground mt-1">Today</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">{taskStats.total}</div>
                <div className="text-sm text-muted-foreground mt-1">Total</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Hardware Stats Card */}
      {hardwareStats && (
        <Card className="mb-8">
            <CardHeader>
              <CardTitle>Hardware Resources</CardTitle>
              <CardDescription>
                {hardwareStats.inference.description}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                {/* CPU & Memory */}
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">CPU Usage</span>
                      <span className="text-sm text-muted-foreground">
                        {hardwareStats.cpu.usage_percent.toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${hardwareStats.cpu.usage_percent}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {hardwareStats.cpu.cores} cores
                      {hardwareStats.cpu.frequency_mhz &&
                        ` @ ${hardwareStats.cpu.frequency_mhz} MHz`
                      }
                    </p>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Memory Usage</span>
                      <span className="text-sm text-muted-foreground">
                        {hardwareStats.memory.used_gb.toFixed(1)} / {hardwareStats.memory.total_gb.toFixed(1)} GB
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
                      <div
                        className="bg-green-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${hardwareStats.memory.usage_percent}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {hardwareStats.memory.usage_percent.toFixed(1)}% used
                    </p>
                  </div>
                </div>

                {/* GPU */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">GPU</span>
                    <Badge variant={hardwareStats.gpu.available ? "default" : "secondary"}>
                      {hardwareStats.gpu.available ? "Available" : "Not Available"}
                    </Badge>
                  </div>
                  {hardwareStats.gpu.available && hardwareStats.gpu.devices.length > 0 ? (
                    <div className="space-y-3">
                      {hardwareStats.gpu.devices.map((gpu) => (
                        <div key={gpu.id} className="p-3 bg-muted/50 rounded-lg space-y-2">
                          <div>
                            <p className="text-sm font-medium">{gpu.name}</p>
                            <p className="text-xs text-muted-foreground">
                              Compute {gpu.compute_capability}
                            </p>
                          </div>
                          <div>
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs">VRAM</span>
                              <span className="text-xs text-muted-foreground">
                                {gpu.reserved_memory_gb.toFixed(1)} / {gpu.total_memory_gb.toFixed(1)} GB
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5 dark:bg-gray-700">
                              <div
                                className="bg-purple-600 h-1.5 rounded-full transition-all duration-300"
                                style={{ width: `${gpu.memory_usage_percent}%` }}
                              ></div>
                            </div>
                          </div>
                          {gpu.utilization_percent !== undefined && (
                            <p className="text-xs text-muted-foreground">
                              Utilization: {gpu.utilization_percent}%
                              {gpu.temperature_c !== undefined &&
                                ` • Temp: ${gpu.temperature_c}°C`
                              }
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      No GPU detected. AI models will run on CPU.
                    </p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
    </div>
  );
}
