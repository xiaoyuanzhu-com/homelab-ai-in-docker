"use client";

import { useState, useEffect, Suspense } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Image as ImageIcon, Loader2, Download } from "lucide-react";
import { TryLayout } from "@/components/try";

type ChoiceType = "model" | "lib";
interface TaskOption {
  id: string;
  label: string;
  provider: string;
  supports_markdown?: boolean;
  status: string;
  type: ChoiceType;
}

interface ConvertResult {
  request_id: string;
  screenshot: string;
  model: string;
  processing_time_ms: number;
}

function DocToScreenshotContent() {
  const [file, setFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ConvertResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [selectedLib, setSelectedLib] = useState<string>("screenitshot/screenitshot");
  const [libsReady, setLibsReady] = useState<boolean>(true);

  useEffect(() => {
    // Check availability via task-options
    const fetchOptions = async () => {
      try {
        const resp = await fetch("/api/task-options?task=doc-to-screenshot");
        if (!resp.ok) throw new Error("Failed to load options");
        const data = await resp.json();
        const ready = (data.options || []).some((o: TaskOption) => o.type === "lib" && o.id === "screenitshot/screenitshot" && o.status === "ready");
        setLibsReady(ready);
        if (!ready) toast.error("Doc to Screenshot engine is not ready");
      } catch (err) {
        console.error(err);
        setLibsReady(false);
      }
    };
    fetchOptions();
  }, []);

  const downloadScreenshot = () => {
    if (!result?.screenshot) return;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${result.screenshot}`;
    link.download = file?.name ? `${file.name.replace(/\.[^.]+$/, "")}_screenshot.png` : "screenshot.png";
    link.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setResult(null);
    setError(null);
    if (f) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setFilePreview(reader.result as string);
      };
      reader.readAsDataURL(f);
    } else {
      setFilePreview("");
    }
  };

  const handleConvert = async () => {
    if (!file || !filePreview) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        file: filePreview,
        filename: file.name,
        lib: selectedLib,
      };
      const resp = await fetch("/api/doc-to-screenshot", {
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
      toast.success("Converted to Screenshot", { description: `Processed in ${data.processing_time_ms}ms` });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to convert document";
      setError(errorMsg);
      toast.error("Conversion failed", { description: errorMsg });
    } finally {
      setLoading(false);
    }
  };

  const requestPayload = {
    lib: selectedLib,
    filename: file?.name || null,
    file: filePreview ? `<base64 data, ${Math.round(filePreview.length / 1024)}KB>` : null,
  };

  const responsePayload = result ? {
    request_id: result.request_id,
    processing_time_ms: result.processing_time_ms,
    model: result.model,
    screenshot: `<base64 PNG, ${Math.round(result.screenshot.length / 1024)}KB>`,
  } : null;

  const inputContent = (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="doc-file">Document</Label>
        <input
          id="doc-file"
          type="file"
          onChange={handleFileChange}
          disabled={loading}
          className="block w-full text-sm text-foreground file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
        />
        <p className="text-xs text-muted-foreground">
          Supported: PDF, DOCX, PPTX, XLSX, EPUB, Markdown, HTML, LaTeX, CSV, and more.
        </p>
      </div>
    </div>
  );

  const outputContent = (
    <div className="space-y-4">
      {error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg text-sm">{error}</div>
      )}

      {result && (
        <>
          <div className="flex items-center justify-end gap-2">
            <Button variant="outline" size="sm" onClick={downloadScreenshot}>
              <Download className="h-4 w-4 mr-2" />
              Download PNG
            </Button>
          </div>

          <div className="space-y-2">
            <Label>Screenshot</Label>
            <div className="border rounded-lg p-4 bg-muted/30">
              <img
                src={`data:image/png;base64,${result.screenshot}`}
                alt="Document screenshot"
                className="max-w-full h-auto rounded shadow-sm"
              />
            </div>
          </div>

          <div className="text-xs text-muted-foreground space-y-1">
            <div>Request ID: {result.request_id}</div>
            <div>Engine: {result.model}</div>
            <div>Processing Time: {result.processing_time_ms}ms</div>
          </div>
        </>
      )}

      {!result && !error && (
        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
          <ImageIcon className="h-12 w-12 mb-2" />
          <p className="text-sm">Upload a document and click Convert</p>
        </div>
      )}
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Doc to Screenshot</h1>
        <p className="text-muted-foreground">Convert documents (PDF, DOCX, PPTX, etc.) to retina-quality PNG screenshots</p>
      </div>

      <TryLayout
        input={{
          title: "Input",
          description: "Upload a document to convert",
          children: inputContent,
          footer: (
            <Button
              onClick={handleConvert}
              disabled={!file || !libsReady || loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Converting...
                </>
              ) : (
                <>
                  <ImageIcon className="mr-2 h-4 w-4" />
                  Convert to Screenshot
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
          description: "Generated Screenshot",
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

export default function DocToScreenshotPage() {
  return (
    <Suspense fallback={<div className="container mx-auto px-4 py-8"><div>Loading...</div></div>}>
      <DocToScreenshotContent />
    </Suspense>
  );
}
