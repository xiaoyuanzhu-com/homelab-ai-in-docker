/**
 * Shared API type definitions for the homelab-ai API
 */

export interface CrawlResult {
  request_id: string;
  url: string;
  title: string | null;
  markdown: string;
  processing_time_ms: number;
  success: boolean;
}

export interface HistoryEntry {
  service: string;
  timestamp: string;
  request_id: string;
  status: string;
  request: Record<string, unknown>;
  response: Record<string, unknown>;
}
