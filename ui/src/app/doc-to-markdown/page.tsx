"use client";

import { useState, useEffect, Suspense } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { FileText, Loader2, Copy } from "lucide-react";
import ReactMarkdown from "react-markdown";
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
  markdown: string;
  model: string;
  processing_time_ms: number;
}

function DocToMarkdownContent() {
  const [file, setFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ConvertResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [selectedLib, setSelectedLib] = useState<string>("microsoft/markitdown");
  const [libsReady, setLibsReady] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<"raw" | "rendered">("rendered");

  useEffect(() => {
    // Check availability via task-options
    const fetchOptions = async () => {
      try {
        const resp = await fetch("/api/task-options?task=doc-to-markdown");
        if (!resp.ok) throw new Error("Failed to load options");
        const data = await resp.json();
        const ready = (data.options || []).some((o: TaskOption) => o.type === "lib" && o.id === "microsoft/markitdown" && o.status === "ready");
        setLibsReady(ready);
        if (!ready) toast.error("Doc to Markdown engine is not ready");
      } catch (err) {
        console.error(err);
        setLibsReady(false);
      }
    };
    fetchOptions();
  }, []);

  const copyToClipboard = async () => {
    if (!result?.markdown) return;
    try {
      await navigator.clipboard.writeText(result.markdown);
      toast.success("Copied to clipboard!");
    } catch {
      toast.error("Failed to copy to clipboard");
    }
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
      const resp = await fetch("/api/doc-to-markdown", {
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
      toast.success("Converted to Markdown", { description: `Processed in ${data.processing_time_ms}ms` });
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

  const responsePayload = result;

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
          Supported: PDF, DOCX, PPTX, XLSX, HTML, TXT, and more.
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

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Markdown</Label>
              <Button variant="ghost" size="sm" onClick={copyToClipboard} className="h-8">
                <Copy className="h-4 w-4 mr-2" />
                Copy
              </Button>
            </div>

            {viewMode === "rendered" ? (
              <div className="border rounded-lg p-4 min-h-[300px] prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown>{result.markdown}</ReactMarkdown>
              </div>
            ) : (
              <Textarea
                value={result.markdown}
                readOnly
                className="min-h-[300px] font-mono text-sm"
                placeholder="Converted markdown will appear here..."
              />
            )}
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
          <FileText className="h-12 w-12 mb-2" />
          <p className="text-sm">Upload a document and click Convert</p>
        </div>
      )}
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Doc to Markdown</h1>
        <p className="text-muted-foreground">Convert documents (PDF, DOCX, PPTX, etc.) to Markdown</p>
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
                  <FileText className="mr-2 h-4 w-4" />
                  Convert to Markdown
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
          description: "Converted Markdown",
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

export default function DocToMarkdownPage() {
  return (
    <Suspense fallback={<div className="container mx-auto px-4 py-8"><div>Loading...</div></div>}>
      <DocToMarkdownContent />
    </Suspense>
  );
}

