/**
 * Task ID to display name mapping
 * Backend uses task IDs, frontend displays user-friendly names
 */

export const TASK_DISPLAY_NAMES: Record<string, string> = {
  "feature-extraction": "Feature Extraction",
  "image-captioning": "Image Captioning",
  "image-ocr": "Image OCR",
};

/**
 * Get display name for a task ID
 */
export function getTaskDisplayName(taskId: string): string {
  return TASK_DISPLAY_NAMES[taskId] || taskId;
}
