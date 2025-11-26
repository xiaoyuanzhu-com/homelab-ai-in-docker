/**
 * Task ID to display name mapping
 * Backend uses task IDs, frontend displays user-friendly names
 */

export const TASK_DISPLAY_NAMES: Record<string, string> = {
  // Text tasks
  "feature-extraction": "Feature Extraction",
  "text-classification": "Text Classification",
  "token-classification": "Token Classification",
  "question-answering": "Question Answering",
  "zero-shot-classification": "Zero Shot Classification",
  "translation": "Translation",
  "summarization": "Summarization",
  "text-generation": "Text Generation",
  "fill-mask": "Fill Mask",
  "sentence-similarity": "Sentence Similarity",

  // Image tasks
  "image-captioning": "Image Captioning",
  "image-ocr": "Image OCR",
  "image-classification": "Image Classification",
  "object-detection": "Object Detection",
  "image-segmentation": "Image Segmentation",
  "text-to-image": "Text to Image",
  "image-to-image": "Image to Image",
  "unconditional-image-generation": "Unconditional Image Generation",
  "zero-shot-image-classification": "Zero Shot Image Classification",

  // Audio tasks
  "audio-classification": "Audio Classification",
  "automatic-speech-recognition": "Automatic Speech Recognition",
  "speaker-embedding": "Speaker Embedding",
  "text-to-speech": "Text to Speech",
  "audio-to-audio": "Audio to Audio",

  // Video tasks
  "video-classification": "Video Classification",
  "video-object-tracking": "Video Object Tracking",
  "video-to-text": "Video to Text",

  // Multimodal tasks
  "document-question-answering": "Document Question Answering",
  "visual-question-answering": "Visual Question Answering",
  "image-text-to-text": "Image Text to Text",
  "table-question-answering": "Table Question Answering",

  // Other tasks
  "web-crawling": "Web Crawling",
  "doc-to-markdown": "Doc to Markdown",
  "tabular-classification": "Tabular Classification",
  "tabular-regression": "Tabular Regression",
  "reinforcement-learning": "Reinforcement Learning",
};

/**
 * Get display name for a task ID
 */
export function getTaskDisplayName(taskId: string): string {
  return TASK_DISPLAY_NAMES[taskId] || taskId;
}
