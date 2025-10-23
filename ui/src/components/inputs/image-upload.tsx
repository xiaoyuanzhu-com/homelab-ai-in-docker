"use client";

import { useCallback, useRef, useState } from "react";
import { Image as ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
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
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleNativeChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      if (disabled) return;
      onChange?.(event);
    },
    [disabled, onChange]
  );

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (disabled) return;

      setIsDragging(false);

      const files = event.dataTransfer?.files;
      if (!files || files.length === 0) return;

      const syntheticEvent = {
        target: { files },
        currentTarget: { files },
      } as unknown as React.ChangeEvent<HTMLInputElement>;

      onChange?.(syntheticEvent);

      if (inputRef.current) {
        inputRef.current.value = "";
      }
    },
    [disabled, onChange]
  );

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (disabled) return;
    setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleClick = useCallback(() => {
    if (disabled) return;
    inputRef.current?.click();
  }, [disabled]);

  const renderDropContent = () => {
    if (previewSrc) {
      return (
        <img
          src={previewSrc}
          alt="Preview"
          className="max-w-full h-auto max-h-96 mx-auto rounded"
        />
      );
    }

    return (
      <div className="flex flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
        <ImageIcon className="h-8 w-8" />
        <span className="font-medium text-foreground">Click to upload</span>
        <span className="text-xs">or drag & drop</span>
      </div>
    );
  };

  return (
    <div className="space-y-2">
      {label ? <Label htmlFor={id}>{label}</Label> : null}

      {!hideInput && (
        <>
          <input
            ref={inputRef}
            id={id}
            type="file"
            accept={accept}
            disabled={disabled}
            onChange={handleNativeChange}
            className="hidden"
          />

          <div
            role="button"
            tabIndex={0}
            onClick={handleClick}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                handleClick();
              }
            }}
            onDragOver={handleDragOver}
            onDragEnter={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={cn(
              "flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-center transition-colors",
              disabled ? "cursor-not-allowed opacity-70" : "cursor-pointer hover:border-primary",
              isDragging ? "border-primary bg-primary/5" : "border-muted"
            )}
          >
            {renderDropContent()}
            {helperText ? (
              <p className="mt-2 text-xs text-muted-foreground">{helperText}</p>
            ) : null}
          </div>
        </>
      )}

      {hideInput && previewSrc ? (
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
