"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Globe,
  Image as ImageIcon,
  Type,
  FileText,
  MessageSquare,
  Languages,
  Camera,
  Volume2,
  Target,
  TrendingUp,
  Layers,
  FileImage,
  Loader2,
  Scan,
  UserCircle,
  Monitor,
  ArrowUpDown,
  Shuffle
} from "lucide-react";

interface Task {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  href?: string;
  available: boolean;
}

interface TaskCategory {
  id: string;
  title: string;
  tasks: Task[];
}

const TASK_CATEGORIES: TaskCategory[] = [
  {
    id: "text",
    title: "Text",
    tasks: [
      {
        id: "text-generation",
        title: "Text Generation",
        description: "Generate text from a prompt",
        icon: <FileText className="h-5 w-5" />,
        href: "/text-generation",
        available: true
      },
      {
        id: "feature-extraction",
        title: "Feature Extraction",
        description: "Extract embeddings from text",
        icon: <Type className="h-5 w-5" />,
        href: "/feature-extraction",
        available: true
      },
      {
        id: "text-ranking",
        title: "Text Ranking",
        description: "Rank and rerank text passages by relevance",
        icon: <ArrowUpDown className="h-5 w-5" />,
        available: false
      },
      {
        id: "translation",
        title: "Translation",
        description: "Translate text between languages",
        icon: <Languages className="h-5 w-5" />,
        available: false
      }
    ]
  },
  {
    id: "image",
    title: "Image",
    tasks: [
      {
        id: "image-captioning",
        title: "Image Captioning",
        description: "Generate text descriptions from images",
        icon: <ImageIcon className="h-5 w-5" />,
        href: "/image-captioning",
        available: true
      },
      {
        id: "image-ocr",
        title: "Image OCR",
        description: "Extract text from images",
        icon: <FileText className="h-5 w-5" />,
        href: "/image-ocr",
        available: true
      },
      {
        id: "object-detection",
        title: "Object Detection",
        description: "Detect objects in images",
        icon: <Target className="h-5 w-5" />,
        available: false
      },
      {
        id: "image-segmentation",
        title: "Image Segmentation",
        description: "Segment images into regions",
        icon: <Layers className="h-5 w-5" />,
        available: false
      },
      {
        id: "face-detection",
        title: "Face Detection",
        description: "Detect faces in images",
        icon: <Scan className="h-5 w-5" />,
        available: false
      },
      {
        id: "face-recognition",
        title: "Face Recognition",
        description: "Identify and verify faces",
        icon: <UserCircle className="h-5 w-5" />,
        available: false
      },
      {
        id: "screen-parsing",
        title: "Screen Parsing",
        description: "Extract structured data from screenshots",
        icon: <Monitor className="h-5 w-5" />,
        available: false
      },
      {
        id: "text-to-image",
        title: "Text to Image",
        description: "Generate images from text",
        icon: <FileImage className="h-5 w-5" />,
        available: false
      },
      {
        id: "image-to-image",
        title: "Image to Image",
        description: "Transform images",
        icon: <Camera className="h-5 w-5" />,
        available: false
      }
    ]
  },
  {
    id: "audio",
    title: "Audio",
    tasks: [
      {
        id: "automatic-speech-recognition",
        title: "Automatic Speech Recognition",
        description: "Transcribe speech to text and identify speakers",
        icon: <Volume2 className="h-5 w-5" />,
        href: "/automatic-speech-recognition",
        available: true
      },
      {
        id: "speaker-embedding",
        title: "Speaker Embedding",
        description: "Extract speaker embeddings for verification",
        icon: <Volume2 className="h-5 w-5" />,
        href: "/speaker-embedding",
        available: true
      },
      {
        id: "text-to-speech",
        title: "Text to Speech",
        description: "Synthesize speech from text",
        icon: <Volume2 className="h-5 w-5" />,
        available: false
      }
    ]
  },
  {
    id: "video",
    title: "Video",
    tasks: [
      {
        id: "video-object-tracking",
        title: "Video Object Tracking",
        description: "Track objects across video frames",
        icon: <Target className="h-5 w-5" />,
        available: false
      },
      {
        id: "video-to-text",
        title: "Video to Text",
        description: "Generate descriptions from videos",
        icon: <FileText className="h-5 w-5" />,
        available: false
      }
    ]
  },
  {
    id: "multimodal",
    title: "Multimodal",
    tasks: [
      {
        id: "any-to-any",
        title: "Any to Any",
        description: "Convert between any modalities: text, image, audio, video",
        icon: <Shuffle className="h-5 w-5" />,
        available: false
      }
    ]
  },
  {
    id: "other",
    title: "Other",
    tasks: [
      {
        id: "web-crawling",
        title: "Web Crawling",
        description: "Extract content from websites",
        icon: <Globe className="h-5 w-5" />,
        href: "/web-crawling",
        available: true
      },
      {
        id: "tabular-regression",
        title: "Tabular Regression",
        description: "Predict continuous values",
        icon: <TrendingUp className="h-5 w-5" />,
        available: false
      }
    ]
  }
];

export default function Home() {
  const [crawlStatus, setCrawlStatus] = useState<"active" | "preparing" | "checking">("checking");

  useEffect(() => {
    // Crawl service is always ready (Playwright pre-installed at build time)
    setCrawlStatus("active");
  }, []);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {TASK_CATEGORIES.map((category) => (
        <div key={category.id} className="mb-12">
          <h2 className="text-2xl font-bold mb-6">{category.title}</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {category.tasks.map((task) => {
              // Special handling for web-crawling task
              const isCrawlTask = task.id === "web-crawling";
              const isTaskAvailable = isCrawlTask
                ? crawlStatus === "active"
                : task.available;
              const isPreparing = isCrawlTask && crawlStatus === "preparing";
              const isChecking = isCrawlTask && crawlStatus === "checking";

              const CardContent = (
                <Card
                  className={`h-full transition-all ${
                    isTaskAvailable
                      ? "hover:shadow-lg cursor-pointer group border-border"
                      : "opacity-60 cursor-not-allowed bg-muted/30"
                  }`}
                >
                  <CardHeader className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div
                        className={`p-2 rounded-lg ${
                          isTaskAvailable
                            ? "bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground"
                            : "bg-muted text-muted-foreground"
                        } transition-colors`}
                      >
                        {isPreparing || isChecking ? (
                          <Loader2 className="h-5 w-5 animate-spin" />
                        ) : (
                          task.icon
                        )}
                      </div>
                      {!task.available && !isCrawlTask && (
                        <Badge variant="secondary" className="text-xs">
                          Coming Soon
                        </Badge>
                      )}
                      {isPreparing && (
                        <Badge variant="secondary" className="text-xs">
                          Preparing...
                        </Badge>
                      )}
                    </div>
                    <CardTitle
                      className={`text-base mb-1 ${
                        isTaskAvailable ? "group-hover:text-primary" : "text-muted-foreground"
                      } transition-colors`}
                    >
                      {task.title}
                    </CardTitle>
                    <CardDescription className="text-xs">
                      {isPreparing
                        ? "Installing Playwright browsers..."
                        : task.description}
                    </CardDescription>
                  </CardHeader>
                </Card>
              );

              return isTaskAvailable && task.href ? (
                <Link key={task.id} href={task.href}>
                  {CardContent}
                </Link>
              ) : (
                <div key={task.id}>{CardContent}</div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
