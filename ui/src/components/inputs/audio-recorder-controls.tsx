"use client";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Mic, StopCircle, Trash2 } from "lucide-react";

interface AudioRecorderControlsProps {
  isRecording: boolean;
  onStart?: () => void;
  onStop?: () => void;
  onClear?: () => void;
  supported?: boolean;
  disabled?: boolean;
}

export function AudioRecorderControls({
  isRecording,
  onStart,
  onStop,
  onClear,
  supported = true,
  disabled,
}: AudioRecorderControlsProps) {
  return (
    <div className="space-y-2">
      <Label>Recording</Label>
      {!supported ? (
        <p className="text-xs text-muted-foreground">
          Recording is not supported in this browser. Use HTTPS and a modern browser to enable microphone access.
        </p>
      ) : (
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant={isRecording ? "destructive" : "default"}
            onClick={() => {
              if (isRecording) {
                onStop?.();
              } else {
                onStart?.();
              }
            }}
            disabled={disabled || (!isRecording && !onStart) || (isRecording && !onStop)}
          >
            {isRecording ? (
              <>
                <StopCircle className="mr-2 h-4 w-4" /> Stop
              </>
            ) : (
              <>
                <Mic className="mr-2 h-4 w-4" /> Record
              </>
            )}
          </Button>
          {onClear && (
            <Button type="button" variant="outline" onClick={onClear} disabled={disabled}>
              <Trash2 className="mr-2 h-4 w-4" /> Clear
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
