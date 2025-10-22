"use client";

import { Label } from "@/components/ui/label";

interface AudioPlayerProps {
  src?: string | null;
  label?: string;
}

export function AudioPlayer({ src, label }: AudioPlayerProps) {
  if (!src) {
    return null;
  }

  return (
    <div className="space-y-2">
      {label ? <Label>{label}</Label> : null}
      <audio controls className="w-full">
        <source src={src} />
        Your browser does not support the audio element.
      </audio>
    </div>
  );
}
