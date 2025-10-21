"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

export default function SettingsPage() {
  const [modelIdleTimeout, setModelIdleTimeout] = useState("5");
  const [loading, setLoading] = useState(true);

  // Load settings on mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const response = await fetch("/api/settings");
        if (response.ok) {
          const data = await response.json();
          if (data.settings.model_idle_timeout_seconds) {
            setModelIdleTimeout(data.settings.model_idle_timeout_seconds);
          }
        }
      } catch (error) {
        console.error("Failed to load settings:", error);
      } finally {
        setLoading(false);
      }
    };
    loadSettings();
  }, []);

  const handleSave = async () => {
    try {
      const response = await fetch("/api/settings/model_idle_timeout_seconds", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value: modelIdleTimeout }),
      });

      if (response.ok) {
        toast.success("Settings saved successfully!");
      } else {
        toast.error("Failed to save settings");
      }
    } catch (error) {
      console.error("Error saving settings:", error);
      toast.error("Failed to save settings");
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Settings</h1>
        <p className="text-muted-foreground">
          Configure application behavior and preferences
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Performance</CardTitle>
          <CardDescription>Optimize GPU memory usage and model loading</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="model-idle-timeout">GPU Model Idle Timeout (seconds)</Label>
            <Input
              id="model-idle-timeout"
              type="number"
              min="1"
              max="300"
              value={modelIdleTimeout}
              onChange={(e) => setModelIdleTimeout(e.target.value)}
              disabled={loading}
            />
            <p className="text-sm text-muted-foreground">
              Time in seconds before unloading idle models from GPU memory. Lower values free GPU
              faster for other services. Higher values improve performance for frequent requests.
              Default: 5 seconds.
            </p>
          </div>

          <Button onClick={handleSave} disabled={loading}>
            Save Settings
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
