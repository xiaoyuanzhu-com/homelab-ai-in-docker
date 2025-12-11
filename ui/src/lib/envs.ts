export interface EnvInfo {
  env_id: string;
  status: "not_installed" | "installing" | "ready" | "failed" | "not_found";
  size_mb: number | null;
  python_version: string | null;
  error_message: string | null;
  install_time_seconds: number | null;
}

export interface EnvsResponse {
  environments: Record<string, EnvInfo>;
}
