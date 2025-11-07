export interface LibInfo {
  id: string;
  label: string;
  provider: string;
  tasks: string[];
  architecture?: string | null;
  default_prompt?: string | null;
  platform_requirements?: string | null;
  supports_markdown?: boolean;
  requires_quantization?: boolean;
  requires_download: boolean;
  status: "init" | "downloading" | "failed" | "ready";
  reference_url?: string | null;
  downloaded_size_mb?: number | null;
  error_message?: string | null;
}

export interface LibsResponse {
  libs: LibInfo[];
}

export const isLibReady = (l: LibInfo): boolean =>
  !l.requires_download || l.status === "ready";

