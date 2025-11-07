"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { FileText, Loader2, Copy } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { TryLayout, ModelSelector } from "@/components/try";
import { ImageUpload } from "@/components/inputs";

interface OCRResult {
  request_id: string;
  text: string;
  model: string;
  processing_time_ms: number;
  output_format: string;
}

type ChoiceType = "model" | "lib";
interface ChoiceInfo {
  id: string;
  label: string;
  provider: string;
  supports_markdown: boolean;
  status: string;
  type: ChoiceType;
}

function ImageOCRContent() {
  const searchParams = useSearchParams();
  const [, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OCRResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedChoice, setSelectedChoice] = useState<string>("");
  const [choices, setChoices] = useState<ChoiceInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [outputFormat, setOutputFormat] = useState<"text" | "markdown">("text");
  const [viewMode, setViewMode] = useState<"raw" | "rendered">("rendered");

  useEffect(() => {
    // Fetch available choices via unified task-options endpoint
    const fetchChoices = async () => {
      try {
        const resp = await fetch("/api/task-options?task=image-ocr");
        if (!resp.ok) throw new Error("Failed to fetch OCR options");
        const data = await resp.json();
        const merged: ChoiceInfo[] = (data.options || [])
          .filter((o: any) => o.status === "ready")
          .map((o: any) => ({ id: o.id, label: o.label, provider: o.provider, supports_markdown: !!o.supports_markdown, status: o.status, type: o.type }));
        setChoices(merged);

        // Preselect logic: prefer explicit model/lib URL params, then legacy skill param, then first available
        const modelParam = searchParams.get("model");
        const libParam = searchParams.get("lib");
        const skillParam = searchParams.get("skill");
        const findChoice = (t: ChoiceType, id: string) => merged.find(c => c.type === t && c.id === id);
        if (modelParam && findChoice("model", modelParam)) {
          setSelectedChoice(`model:${modelParam}`);
        } else if (libParam && findChoice("lib", libParam)) {
          setSelectedChoice(`lib:${libParam}`);
        } else if (skillParam) {
          const m = findChoice("model", skillParam);
          const l = findChoice("lib", skillParam);
          if (m) setSelectedChoice(`model:${skillParam}`);
          else if (l) setSelectedChoice(`lib:${skillParam}`);
          else if (merged.length > 0) setSelectedChoice(`${merged[0].type}:${merged[0].id}`);
        } else if (merged.length > 0) {
          setSelectedChoice(`${merged[0].type}:${merged[0].id}`);
        }
      } catch (err) {
        console.error("Error fetching OCR choices:", err);
        toast.error("Failed to load available OCR engines");
      } finally {
        setModelsLoading(false);
      }
    };

    fetchChoices();
  }, [searchParams]);

  // Check if selected model supports markdown
  const supportsMarkdown = () => {
    if (!selectedChoice) return false;
    const [type, id] = selectedChoice.split(":", 2) as [ChoiceType, string];
    const choice = choices.find((m) => m.type === type && m.id === id);
    if (!choice) return false;
    return choice.supports_markdown === true;
  };

  const copyToClipboard = async () => {
    if (!result?.text) return;
    try {
      await navigator.clipboard.writeText(result.text);
      toast.success("Copied to clipboard!");
    } catch {
      toast.error("Failed to copy to clipboard");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleOCR = async () => {
    if (!imagePreview || !selectedChoice) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Extract base64 from data URL
      const base64Data = imagePreview.split(",")[1];

      const response = await fetch("/api/image-ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: base64Data,
          ...(selectedChoice.startsWith("model:")
            ? { model: selectedChoice.split(":", 2)[1] }
            : { lib: selectedChoice.split(":", 2)[1] }),
          output_format: outputFormat,
        }),
      });

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errorData = await response.json();
          if (errorData.detail?.message) {
            errorMessage = errorData.detail.message;
          } else if (typeof errorData.detail === 'string') {
            errorMessage = errorData.detail;
          }
        } catch {
          // If JSON parsing fails, use the default error message
          console.error('Failed to parse error response as JSON');
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      setResult(data);
      toast.success("Text extracted successfully!", {
        description: `Processed in ${data.processing_time_ms}ms`,
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to extract text";
      setError(errorMsg);
      toast.error("OCR failed", {
        description: errorMsg,
      });
    } finally {
      setLoading(false);
    }
  };

  const modelOptions = choices.map((c) => ({
    value: `${c.type}:${c.id}`,
    label: `${c.label} (${c.provider}) [${c.type}]`,
  }));

  const requestPayload = {
    ...(selectedChoice?.startsWith("model:")
      ? { model: selectedChoice.split(":", 2)[1] }
      : selectedChoice
      ? { lib: selectedChoice.split(":", 2)[1] }
      : { model: null }),
    output_format: outputFormat,
    image: imagePreview ? `<base64 data, ${Math.round(imagePreview.length / 1024)}KB>` : null,
  };

  const responsePayload = result;

  const inputContent = (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="model">Engine</Label>
        <ModelSelector
          value={selectedChoice}
          onChange={setSelectedChoice}
          options={modelOptions}
          loading={modelsLoading}
          disabled={loading}
          emptyMessage="No OCR engines ready. Visit Models/Libs to set them up."
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="format">Output Format</Label>
        <Select
          value={outputFormat}
          onValueChange={(value) => setOutputFormat(value as "text" | "markdown")}
          disabled={!supportsMarkdown()}
        >
          <SelectTrigger id="format">
            <SelectValue placeholder="Select format" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="text">Plain Text</SelectItem>
            <SelectItem value="markdown">Markdown</SelectItem>
          </SelectContent>
        </Select>
        {!supportsMarkdown() && selectedModel && (
          <p className="text-xs text-muted-foreground">
            This model only supports plain text output.
          </p>
        )}
        {supportsMarkdown() && (
          <p className="text-xs text-muted-foreground">Markdown rendering supported for this model.</p>
        )}
      </div>

      <ImageUpload
        id="ocr-image"
        label="Image"
        onChange={handleFileChange}
        previewSrc={imagePreview}
        disabled={loading}
        helperText="Supported formats: PNG, JPG, WebP."
      />
    </div>
  );

  const outputContent = (
    <div className="space-y-4">
      {error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg text-sm">{error}</div>
      )}

      {result && (
        <>
          {result.output_format === "markdown" && (
            <div className="flex items-center justify-end gap-2">
              <Button
                variant={viewMode === "rendered" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("rendered")}
              >
                Rendered
              </Button>
              <Button
                variant={viewMode === "raw" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("raw")}
              >
                Raw
              </Button>
            </div>
          )}

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Extracted Text</Label>
              <Button variant="ghost" size="sm" onClick={copyToClipboard} className="h-8">
                <Copy className="h-4 w-4 mr-2" />
                Copy
              </Button>
            </div>

            {result.output_format === "markdown" && viewMode === "rendered" ? (
              <div className="border rounded-lg p-4 min-h-[300px] prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown>{result.text}</ReactMarkdown>
              </div>
            ) : (
              <Textarea
                value={result.text}
                readOnly
                className="min-h-[300px] font-mono text-sm"
                placeholder="Extracted text will appear here..."
              />
            )}
          </div>

          <div className="text-xs text-muted-foreground space-y-1">
            <div>Request ID: {result.request_id}</div>
            <div>Model: {result.model}</div>
            <div>Output Format: {result.output_format}</div>
            <div>Processing Time: {result.processing_time_ms}ms</div>
          </div>
        </>
      )}

      {!result && !error && (
        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
          <FileText className="h-12 w-12 mb-2" />
          <p className="text-sm">Upload an image and click Extract Text</p>
        </div>
      )}
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Image OCR</h1>
        <p className="text-muted-foreground">
          Extract text from images using optical character recognition
        </p>
      </div>

      <TryLayout
        input={{
          title: "Input",
          description: "Upload an image to extract text",
          children: inputContent,
          footer: (
            <Button
              onClick={handleOCR}
              disabled={!imagePreview || !selectedChoice || loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Extracting...
                </>
              ) : (
                <>
                  <FileText className="mr-2 h-4 w-4" />
                  Extract Text
                </>
              )}
            </Button>
          ),
          rawPayload: {
            label: "View Raw Request",
            payload: requestPayload,
          },
        }}
        output={{
          title: "Output",
          description: "Extracted text from the image",
          children: outputContent,
          rawPayload: {
            label: "View Raw Response",
            payload: responsePayload,
          },
        }}
      />
    </div>
  );
}

export default function ImageOCRPage() {
  return (
    <Suspense fallback={<div className="container mx-auto px-4 py-8"><div>Loading...</div></div>}>
      <ImageOCRContent />
    </Suspense>
  );
}
