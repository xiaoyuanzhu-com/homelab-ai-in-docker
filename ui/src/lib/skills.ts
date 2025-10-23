export interface SkillInfo {
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
  status: "init" | "downloading" | "failed" | "downloaded";
  hf_model?: string | null;
  reference_url?: string | null;
  size_mb?: number | null;
  parameters_m?: number | null;
  gpu_memory_mb?: number | null;
  downloaded_size_mb?: number | null;
  error_message?: string | null;
}

export interface SkillsResponse {
  skills: SkillInfo[];
}

export const isSkillReady = (skill: SkillInfo): boolean =>
  !skill.requires_download || skill.status === "downloaded";
