"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { TryLayout } from "@/components/try";
import { ImageUpload } from "@/components/inputs";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Loader2 } from "lucide-react";

type ChoiceType = "model" | "lib";

interface TaskOption {
  id: string;
  label: string;
  provider: string;
  status: string;
  type: ChoiceType;
}

interface SegmentationMask {
  rle: { size: number[]; counts: number[] };
  score?: number | null;
  box?: number[] | null;
}

interface SegmentationResult {
  request_id: string;
  processing_time_ms: number;
  model: string;
  prompt: string;
  image_width: number;
  image_height: number;
  masks: SegmentationMask[];
}

const MASK_COLOR: [number, number, number, number] = [14, 165, 233, 140];

const buildMaskImageData = (
  rle: SegmentationMask["rle"],
  color: [number, number, number, number] = MASK_COLOR
) => {
  if (!rle?.size || rle.size.length < 2) {
    return new ImageData(1, 1);
  }
  const [height, width] = rle.size;
  const imageData = new ImageData(width, height);
  const data = imageData.data;
  let idx = 0;
  let value = 0;

  for (const run of rle.counts) {
    if (value === 1) {
      for (let i = 0; i < run; i += 1) {
        const pos = idx + i;
        const row = pos % height;
        const col = Math.floor(pos / height);
        const pixelIndex = (row * width + col) * 4;
        data[pixelIndex] = color[0];
        data[pixelIndex + 1] = color[1];
        data[pixelIndex + 2] = color[2];
        data[pixelIndex + 3] = color[3];
      }
    }
    idx += run;
    value = 1 - value;
  }

  return imageData;
};

export default function SegmentAnythingPage() {
  const [imagePreview, setImagePreview] = useState<string>("");
  const [prompt, setPrompt] = useState<string>("");
  const [confidenceThreshold, setConfidenceThreshold] = useState<string>("0.5");
  const [maxMasks, setMaxMasks] = useState<string>("8");
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [libsReady, setLibsReady] = useState(true);
  const [selectedLib] = useState("facebookresearch/sam3");
  const [hoveredMask, setHoveredMask] = useState<number | null>(null);
  const [overlaySize, setOverlaySize] = useState({ width: 0, height: 0 });
  const previewImageRef = useRef<HTMLImageElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const resp = await fetch("/api/task-options?task=image-segmentation");
        if (!resp.ok) throw new Error("Failed to load segmentation options");
        const data = await resp.json();
        const ready = (data.options || []).some(
          (o: TaskOption) =>
            o.type === "lib" && o.id === "facebookresearch/sam3" && o.status === "ready"
        );
        setLibsReady(ready);
        if (!ready) toast.error("SAM engine is not ready");
      } catch (err) {
        console.error("Failed to fetch segmentation options:", err);
        setLibsReady(false);
      }
    };

    fetchOptions();
  }, []);

  useEffect(() => {
    const imageEl = previewImageRef.current;
    if (!imageEl) return;

    const updateSize = () => {
      const width = Math.round(imageEl.clientWidth);
      const height = Math.round(imageEl.clientHeight);
      if (width > 0 && height > 0) {
        setOverlaySize({ width, height });
      }
    };

    updateSize();

    const observer = new ResizeObserver(() => updateSize());
    observer.observe(imageEl);
    return () => observer.disconnect();
  }, [imagePreview]);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = overlaySize;
    if (!width || !height) return;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    if (!result || hoveredMask === null) return;
    const mask = result.masks[hoveredMask];
    if (!mask) return;

    const [maskHeight, maskWidth] = mask.rle.size;
    const offscreen = document.createElement("canvas");
    offscreen.width = maskWidth;
    offscreen.height = maskHeight;
    const offscreenCtx = offscreen.getContext("2d");
    if (!offscreenCtx) return;

    offscreenCtx.putImageData(buildMaskImageData(mask.rle), 0, 0);
    ctx.drawImage(offscreen, 0, 0, width, height);
  }, [hoveredMask, overlaySize, result]);

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleRun = async () => {
    if (!imagePreview) return;
    setLoading(true);
    setResult(null);
    setError(null);

    const threshold =
      confidenceThreshold.trim() === "" ? 0.5 : Number(confidenceThreshold);
    const maxMasksValue = maxMasks.trim() ? Number.parseInt(maxMasks, 10) : null;

    try {
      const payload = {
        image: imagePreview,
        lib: selectedLib,
        confidence_threshold: Number.isFinite(threshold) ? threshold : 0.5,
        max_masks: Number.isFinite(maxMasksValue) ? maxMasksValue : null,
        ...(prompt.trim() ? { prompt } : {}),
      };

      const resp = await fetch("/api/sam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        let msg = `HTTP ${resp.status}: ${resp.statusText}`;
        try {
          const j = await resp.json();
          if (j?.detail?.message) msg = j.detail.message;
        } catch {}
        throw new Error(msg);
      }

      const data = await resp.json();
      setResult(data);
      toast.success("Segmentation complete", {
        description: `Found ${data.masks?.length || 0} masks in ${data.processing_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to segment image";
      setError(errorMsg);
      toast.error("Segmentation failed", { description: errorMsg });
    } finally {
      setLoading(false);
    }
  };

  const requestPayload = {
    lib: selectedLib,
    prompt: prompt.trim() ? prompt : null,
    confidence_threshold: confidenceThreshold,
    max_masks: maxMasks || null,
    image: imagePreview ? `<base64 data, ${Math.round(imagePreview.length / 1024)}KB>` : null,
  };

  const outputContent = result ? (
    <div className="space-y-4">
      <div className="text-sm text-muted-foreground">
        <p>Model: {result.model}</p>
        <p>Prompt: {result.prompt}</p>
        <p>Masks: {result.masks.length}</p>
      </div>
      {result.masks.length > 0 ? (
        <div className="space-y-2 max-h-[520px] overflow-auto pr-1">
          {result.masks.map((mask, idx) => (
            <div
              key={`${idx}-${mask.score ?? "mask"}`}
              className={`rounded-lg border px-3 py-2 text-xs transition ${
                hoveredMask === idx ? "border-sky-500 bg-sky-500/10" : "bg-muted/20"
              }`}
              onMouseEnter={() => setHoveredMask(idx)}
              onMouseLeave={() => setHoveredMask(null)}
            >
              <div className="flex items-center justify-between text-sm font-medium">
                <span>Mask {idx + 1}</span>
                <span>{mask.score?.toFixed(3) ?? "n/a"}</span>
              </div>
              <div className="mt-1 text-muted-foreground">
                Size: {mask.rle.size?.[1]}x{mask.rle.size?.[0]}
              </div>
              <div className="text-muted-foreground">
                Counts: {mask.rle.counts.length}
              </div>
              <div className="mt-1 text-[11px] text-muted-foreground/80">
                {mask.rle.counts.slice(0, 12).join(", ")}
                {mask.rle.counts.length > 12 ? " …" : ""}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">No masks returned.</p>
      )}
    </div>
  ) : error ? (
    <div className="text-sm text-destructive">{error}</div>
  ) : (
    <div className="text-sm text-muted-foreground">
      Upload an image and run SAM to see masks.
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">SAM</h1>
        <p className="text-muted-foreground">
          Promptable image segmentation powered by SAM3.
        </p>
      </div>

      <TryLayout
        input={{
          title: "Input",
          description: "Upload an image; hover a mask in the output to preview it here.",
          children: (
            <div className="space-y-4">
              <ImageUpload
                label="Image"
                onChange={handleImageChange}
                helperText="Supports JPG, PNG, WEBP."
              />
              {imagePreview ? (
                <div className="relative overflow-hidden rounded-lg border bg-muted/20">
                  <img
                    ref={previewImageRef}
                    src={imagePreview}
                    alt="Segmentation input"
                    className="block w-full h-auto"
                    onLoad={() => {
                      const imageEl = previewImageRef.current;
                      if (!imageEl) return;
                      const width = Math.round(imageEl.clientWidth);
                      const height = Math.round(imageEl.clientHeight);
                      if (width > 0 && height > 0) {
                        setOverlaySize({ width, height });
                      }
                    }}
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    className="absolute inset-0 pointer-events-none"
                  />
                </div>
              ) : null}
              <div className="space-y-2">
                <Label htmlFor="sam3-prompt">Prompt</Label>
                <Textarea
                  id="sam3-prompt"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  placeholder="Leave blank to auto-segment everything"
                  rows={3}
                />
                <p className="text-xs text-muted-foreground">
                  Example: &quot;person&quot; or &quot;car&quot;. Leave blank for automatic mask generation.
                </p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="sam3-threshold">Confidence Threshold</Label>
                  <Input
                    id="sam3-threshold"
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    value={confidenceThreshold}
                    onChange={(event) => setConfidenceThreshold(event.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="sam3-max-masks">Max Masks</Label>
                  <Input
                    id="sam3-max-masks"
                    type="number"
                    step="1"
                    min="1"
                    value={maxMasks}
                    onChange={(event) => setMaxMasks(event.target.value)}
                  />
                </div>
              </div>
              <Button onClick={handleRun} disabled={!imagePreview || !libsReady || loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Running...
                  </>
                ) : (
                  "Run SAM"
                )}
              </Button>
            </div>
          ),
          rawPayload: {
            label: "Request payload",
            payload: requestPayload,
            emptyMessage: "Upload an image to see the request payload.",
          },
        }}
        output={{
          title: "Output",
          description: "Segmentation masks from SAM3.",
          children: outputContent,
          rawPayload: {
            label: "Response payload",
            payload: result,
            emptyMessage: "Run SAM to see the response payload.",
          },
        }}
      />
    </div>
  );
}
