"use client";

import { Label } from "@/components/ui/label";

interface AudioUploadProps {
  id?: string;
  label?: string;
  fileName?: string;
  onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void;
  disabled?: boolean;
  accept?: string;
  helperText?: string;
}

export function AudioUpload({
  id = "audio-upload",
  label,
  fileName,
  onChange,
  disabled,
  accept = "audio/*",
  helperText,
}: AudioUploadProps) {
  return (
    <div className="space-y-2">
      {label ? <Label htmlFor={id}>{label}</Label> : null}
      <input
        id={id}
        type="file"
        accept={accept}
        onChange={onChange}
        disabled={disabled}
        className="block w-full text-sm text-muted-foreground file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90 disabled:cursor-not-allowed"
      />
      {fileName ? (
        <p className="text-xs text-muted-foreground">Selected: {fileName}</p>
      ) : null}
      {helperText ? (
        <p className="text-xs text-muted-foreground">{helperText}</p>
      ) : null}
    </div>
  );
}
