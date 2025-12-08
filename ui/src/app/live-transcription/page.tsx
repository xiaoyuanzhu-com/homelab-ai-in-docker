"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Mic, MicOff, Loader2, Radio } from "lucide-react";
import { toast } from "sonner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

// WhisperLiveKit Segment format from FrontData.to_dict()
interface TranscriptionSegment {
  speaker: number;
  text: string;
  start: string;  // Formatted as "HH:MM:SS"
  end: string;    // Formatted as "HH:MM:SS"
  translation?: string;
  detected_language?: string;
}

// WhisperLiveKit response format
interface TranscriptionMessage {
  type?: string;
  status?: string;
  lines?: TranscriptionSegment[];  // Array of segment objects
  buffer_transcription?: string;
  buffer_diarization?: string;
  buffer_translation?: string;
  remaining_time_transcription?: number;
  remaining_time_diarization?: number;
  error?: string;
  useAudioWorklet?: boolean;
}

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

export default function LiveTranscriptionPage() {
  // WebSocket and connection state
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Configuration
  const [model, setModel] = useState<string>("large-v3");
  const [language, setLanguage] = useState<string>("en");

  // Transcription results
  const [lines, setLines] = useState<TranscriptionSegment[]>([]);
  const [buffer, setBuffer] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // Audio recording config
  const chunkDuration = 100; // ms

  // Check if browser supports media recording
  const [isSupported, setIsSupported] = useState(false);
  useEffect(() => {
    if (typeof window !== "undefined") {
      setIsSupported(!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia));
    }
  }, []);

  // Build WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const params = new URLSearchParams();
    params.set("model", model);
    params.set("language", language);
    return `${protocol}//${host}/api/live-transcription?${params.toString()}`;
  }, [model, language]);

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data: TranscriptionMessage = JSON.parse(event.data);

      if (data.type === "config") {
        // Server config message (useAudioWorklet flag)
        console.log("Received config:", data);
        return;
      }

      if (data.type === "ready_to_stop") {
        // Server is done processing
        console.log("Server ready to stop");
        return;
      }

      // Update status
      if (data.status) {
        setStatus(data.status);
      }

      // Update transcription lines (array of segment objects)
      if (data.lines !== undefined && Array.isArray(data.lines)) {
        setLines(data.lines);
      }

      // Update buffer (partial transcription in progress)
      if (data.buffer_transcription !== undefined) {
        setBuffer(data.buffer_transcription);
      }

      // Handle errors from server
      if (data.error) {
        setError(data.error);
      }
    } catch (err) {
      console.error("Failed to parse message:", err);
    }
  }, []);

  // Start streaming
  const startStreaming = useCallback(async () => {
    if (!isSupported) {
      toast.error("Your browser doesn't support audio recording");
      return;
    }

    setError(null);
    setLines([]);
    setBuffer("");
    setStatus("");
    setConnectionStatus("connecting");

    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Connect to WebSocket
      const wsUrl = getWebSocketUrl();
      console.log("Connecting to:", wsUrl);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connected");
        setConnectionStatus("connected");

        // Start MediaRecorder
        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: "audio/webm;codecs=opus",
        });
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            ws.send(e.data);
          }
        };

        mediaRecorder.start(chunkDuration);
        setIsStreaming(true);
        toast.success("Live transcription started");
      };

      ws.onmessage = handleMessage;

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        setError("WebSocket connection error");
        setConnectionStatus("error");
      };

      ws.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason);
        setConnectionStatus("disconnected");
        setIsStreaming(false);

        // Clean up media recorder
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
        }
      };
    } catch (err) {
      console.error("Failed to start streaming:", err);
      const errorMsg = err instanceof Error ? err.message : "Failed to start streaming";
      setError(errorMsg);
      setConnectionStatus("error");
      toast.error("Failed to start", { description: errorMsg });
    }
  }, [isSupported, getWebSocketUrl, handleMessage]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    // Stop MediaRecorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }

    // Stop all audio tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsStreaming(false);
    setConnectionStatus("disconnected");
    toast.info("Live transcription stopped");
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  // Get status badge variant
  const getStatusBadge = () => {
    switch (connectionStatus) {
      case "connected":
        return <Badge variant="default" className="bg-green-500">Connected</Badge>;
      case "connecting":
        return <Badge variant="secondary">Connecting...</Badge>;
      case "error":
        return <Badge variant="destructive">Error</Badge>;
      default:
        return <Badge variant="outline">Disconnected</Badge>;
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Radio className="h-8 w-8" />
          Live Transcription
        </h1>
        <p className="text-muted-foreground">
          Real-time speech-to-text transcription using WhisperLiveKit
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[400px_1fr]">
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>
              Configure the live transcription settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Connection Status */}
            <div className="flex items-center justify-between">
              <Label>Status</Label>
              {getStatusBadge()}
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label htmlFor="model">Whisper Model</Label>
              <Select
                value={model}
                onValueChange={setModel}
                disabled={isStreaming}
              >
                <SelectTrigger id="model">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="tiny">tiny (fastest)</SelectItem>
                  <SelectItem value="base">base</SelectItem>
                  <SelectItem value="small">small</SelectItem>
                  <SelectItem value="medium">medium</SelectItem>
                  <SelectItem value="large-v2">large-v2</SelectItem>
                  <SelectItem value="large-v3">large-v3 (best quality)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Language Selection */}
            <div className="space-y-2">
              <Label htmlFor="language">Language</Label>
              <Select
                value={language}
                onValueChange={setLanguage}
                disabled={isStreaming}
              >
                <SelectTrigger id="language">
                  <SelectValue placeholder="Select language" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect</SelectItem>
                  <SelectItem value="en">English</SelectItem>
                  <SelectItem value="zh">Chinese</SelectItem>
                  <SelectItem value="es">Spanish</SelectItem>
                  <SelectItem value="fr">French</SelectItem>
                  <SelectItem value="de">German</SelectItem>
                  <SelectItem value="ja">Japanese</SelectItem>
                  <SelectItem value="ko">Korean</SelectItem>
                  <SelectItem value="ru">Russian</SelectItem>
                  <SelectItem value="pt">Portuguese</SelectItem>
                  <SelectItem value="it">Italian</SelectItem>
                  <SelectItem value="nl">Dutch</SelectItem>
                  <SelectItem value="ar">Arabic</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Microphone Support Warning */}
            {!isSupported && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Not Supported</AlertTitle>
                <AlertDescription>
                  Your browser does not support audio recording. Please use a modern browser with HTTPS.
                </AlertDescription>
              </Alert>
            )}

            {/* Error Display */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Start/Stop Button */}
            <Button
              onClick={isStreaming ? stopStreaming : startStreaming}
              disabled={!isSupported || connectionStatus === "connecting"}
              className="w-full"
              variant={isStreaming ? "destructive" : "default"}
            >
              {connectionStatus === "connecting" ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Connecting...
                </>
              ) : isStreaming ? (
                <>
                  <MicOff className="h-4 w-4 mr-2" />
                  Stop Transcription
                </>
              ) : (
                <>
                  <Mic className="h-4 w-4 mr-2" />
                  Start Transcription
                </>
              )}
            </Button>

            {/* Recording Indicator */}
            {isStreaming && (
              <div className="flex items-center gap-2 p-4 bg-destructive/10 text-destructive rounded-lg">
                <div className="h-3 w-3 bg-destructive rounded-full animate-pulse" />
                <span className="text-sm font-medium">Listening...</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Transcription Output Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Transcription</CardTitle>
            <CardDescription>
              Real-time transcription results will appear here
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[500px] w-full rounded-md border p-4">
              {lines.length === 0 && !buffer && (
                <p className="text-muted-foreground text-center py-8">
                  {isStreaming
                    ? "Speak into your microphone..."
                    : "Start transcription to see results here."}
                </p>
              )}

              {/* Finalized lines */}
              <div className="space-y-2">
                {lines.map((segment, idx) => (
                  <div key={idx} className="p-2 rounded bg-muted/50">
                    {segment.speaker > 0 && (
                      <Badge variant="secondary" className="mr-2 text-xs">
                        Speaker {segment.speaker}
                      </Badge>
                    )}
                    <span>{segment.text}</span>
                    {segment.start && segment.end && (
                      <span className="text-xs text-muted-foreground ml-2">
                        ({segment.start} - {segment.end})
                      </span>
                    )}
                    {segment.detected_language && (
                      <Badge variant="outline" className="ml-2 text-xs">
                        {segment.detected_language}
                      </Badge>
                    )}
                  </div>
                ))}
              </div>

              {/* Buffer (partial/in-progress transcription) */}
              {buffer && (
                <div className="p-2 rounded bg-yellow-100/50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 mt-2">
                  <span className="text-muted-foreground italic">{buffer}</span>
                </div>
              )}
            </ScrollArea>

            {/* Stats */}
            {(lines.length > 0 || buffer) && (
              <div className="mt-4 flex gap-4 text-sm text-muted-foreground">
                {status && <span>Status: {status}</span>}
                <span>Segments: {lines.length}</span>
                <span>
                  Characters: {lines.reduce((acc, s) => acc + (s.text?.length || 0), 0) + buffer.length}
                </span>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
