"use client";

import { Label } from "@/components/ui/label";

interface ImageUploadProps {
  id?: string;
  label?: string;
  previewSrc?: string;
  onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void;
  disabled?: boolean;
  accept?: string;
  helperText?: string;
  hideInput?: boolean;
}

export function ImageUpload({
  id = "image-upload",
  label,
  previewSrc,
  onChange,
  disabled,
  accept = "image/*",
  helperText,
  hideInput,
}: ImageUploadProps) {
  return (
    <div className="space-y-2">
      {label ? <Label htmlFor={id}>{label}</Label> : null}
      {!hideInput && (
        <>
          <input
            id={id}
            type="file"
            accept={accept}
            onChange={onChange}
            disabled={disabled}
            className="block w-full text-sm text-muted-foreground file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90 disabled:cursor-not-allowed"
          />
          {helperText ? (
            <p className="text-xs text-muted-foreground">{helperText}</p>
          ) : null}
        </>
      )}
      {previewSrc ? (
        <div className="border rounded-lg p-2 bg-muted/50">
          <img
            src={previewSrc}
            alt="Preview"
            className="max-w-full h-auto max-h-96 mx-auto rounded"
          />
        </div>
      ) : null}
    </div>
  );
}
