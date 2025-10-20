"use client";

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
  Video,
  Volume2,
  Target,
  Brain,
  TrendingUp,
  Table2,
  Search,
  GitBranch,
  Layers,
  FileImage,
  Boxes,
  Zap
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
        id: "feature-extraction",
        title: "Feature Extraction",
        description: "Extract embeddings from text",
        icon: <Type className="h-5 w-5" />,
        href: "/embedding",
        available: true
      },
      {
        id: "text-classification",
        title: "Text Classification",
        description: "Assign categories to text data",
        icon: <FileText className="h-5 w-5" />,
        available: false
      },
      {
        id: "token-classification",
        title: "Token Classification",
        description: "Assign labels to individual tokens",
        icon: <Target className="h-5 w-5" />,
        available: false
      },
      {
        id: "question-answering",
        title: "Question Answering",
        description: "Answer questions from a context",
        icon: <MessageSquare className="h-5 w-5" />,
        available: false
      },
      {
        id: "zero-shot-classification",
        title: "Zero Shot Classification",
        description: "Classify text without training data",
        icon: <Zap className="h-5 w-5" />,
        available: false
      },
      {
        id: "translation",
        title: "Translation",
        description: "Translate text between languages",
        icon: <Languages className="h-5 w-5" />,
        available: false
      },
      {
        id: "summarization",
        title: "Summarization",
        description: "Create summaries of long documents",
        icon: <FileText className="h-5 w-5" />,
        available: false
      },
      {
        id: "text-generation",
        title: "Text Generation",
        description: "Generate text from a prompt",
        icon: <FileText className="h-5 w-5" />,
        available: false
      },
      {
        id: "fill-mask",
        title: "Fill Mask",
        description: "Predict masked words in text",
        icon: <Brain className="h-5 w-5" />,
        available: false
      },
      {
        id: "sentence-similarity",
        title: "Sentence Similarity",
        description: "Compare sentence similarity",
        icon: <GitBranch className="h-5 w-5" />,
        available: false
      }
    ]
  },
  {
    id: "image",
    title: "Image",
    tasks: [
      {
        id: "image-to-text",
        title: "Image to Text",
        description: "Generate text descriptions from images",
        icon: <ImageIcon className="h-5 w-5" />,
        href: "/image-to-text",
        available: true
      },
      {
        id: "image-classification",
        title: "Image Classification",
        description: "Assign labels to images",
        icon: <ImageIcon className="h-5 w-5" />,
        available: false
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
      },
      {
        id: "unconditional-image-generation",
        title: "Unconditional Image Generation",
        description: "Generate images without conditioning",
        icon: <Boxes className="h-5 w-5" />,
        available: false
      },
      {
        id: "zero-shot-image-classification",
        title: "Zero Shot Image Classification",
        description: "Classify images without training",
        icon: <Zap className="h-5 w-5" />,
        available: false
      }
    ]
  },
  {
    id: "audio",
    title: "Audio",
    tasks: [
      {
        id: "audio-classification",
        title: "Audio Classification",
        description: "Classify audio recordings",
        icon: <Volume2 className="h-5 w-5" />,
        available: false
      },
      {
        id: "automatic-speech-recognition",
        title: "Automatic Speech Recognition",
        description: "Transcribe speech to text",
        icon: <Volume2 className="h-5 w-5" />,
        available: false
      },
      {
        id: "text-to-speech",
        title: "Text to Speech",
        description: "Synthesize speech from text",
        icon: <Volume2 className="h-5 w-5" />,
        available: false
      },
      {
        id: "audio-to-audio",
        title: "Audio to Audio",
        description: "Transform audio",
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
        id: "video-classification",
        title: "Video Classification",
        description: "Classify videos",
        icon: <Video className="h-5 w-5" />,
        available: false
      },
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
        id: "document-question-answering",
        title: "Document Question Answering",
        description: "Answer questions about documents",
        icon: <FileText className="h-5 w-5" />,
        available: false
      },
      {
        id: "visual-question-answering",
        title: "Visual Question Answering",
        description: "Answer questions about images",
        icon: <MessageSquare className="h-5 w-5" />,
        available: false
      },
      {
        id: "image-text-to-text",
        title: "Image Text to Text",
        description: "Generate text from image and text",
        icon: <FileImage className="h-5 w-5" />,
        available: false
      },
      {
        id: "table-question-answering",
        title: "Table Question Answering",
        description: "Answer questions about tabular data",
        icon: <Table2 className="h-5 w-5" />,
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
        href: "/crawl",
        available: true
      },
      {
        id: "tabular-classification",
        title: "Tabular Classification",
        description: "Classify tabular data",
        icon: <Table2 className="h-5 w-5" />,
        available: false
      },
      {
        id: "tabular-regression",
        title: "Tabular Regression",
        description: "Predict continuous values",
        icon: <TrendingUp className="h-5 w-5" />,
        available: false
      },
      {
        id: "reinforcement-learning",
        title: "Reinforcement Learning",
        description: "Train agents through interaction",
        icon: <Brain className="h-5 w-5" />,
        available: false
      }
    ]
  }
];

export default function Home() {
  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {TASK_CATEGORIES.map((category) => (
        <div key={category.id} className="mb-12">
          <h2 className="text-2xl font-bold mb-6">{category.title}</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {category.tasks.map((task) => {
              const CardContent = (
                <Card
                  className={`h-full transition-all ${
                    task.available
                      ? "hover:shadow-lg cursor-pointer group border-border"
                      : "opacity-60 cursor-not-allowed bg-muted/30"
                  }`}
                >
                  <CardHeader className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div
                        className={`p-2 rounded-lg ${
                          task.available
                            ? "bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground"
                            : "bg-muted text-muted-foreground"
                        } transition-colors`}
                      >
                        {task.icon}
                      </div>
                      {!task.available && (
                        <Badge variant="secondary" className="text-xs">
                          Coming Soon
                        </Badge>
                      )}
                    </div>
                    <CardTitle
                      className={`text-base mb-1 ${
                        task.available ? "group-hover:text-primary" : "text-muted-foreground"
                      } transition-colors`}
                    >
                      {task.title}
                    </CardTitle>
                    <CardDescription className="text-xs">{task.description}</CardDescription>
                  </CardHeader>
                </Card>
              );

              return task.available && task.href ? (
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
