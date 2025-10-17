"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownDocProps {
  content: string;
  apiBaseUrl?: string;
}

export function MarkdownDoc({ content, apiBaseUrl = "" }: MarkdownDocProps) {
  // Replace {{API_BASE_URL}} placeholder with actual URL
  const processedContent = content.replace(/\{\{API_BASE_URL\}\}/g, apiBaseUrl);

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-2xl font-bold mb-2">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xl font-semibold mb-2 mt-6">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-lg font-semibold mb-2 mt-4">{children}</h3>
          ),
          p: ({ children }) => (
            <p className="text-sm mb-4">{children}</p>
          ),
          ul: ({ children }) => (
            <ul className="space-y-2 text-sm mb-4">{children}</ul>
          ),
          li: ({ children }) => (
            <li className="ml-4">{children}</li>
          ),
          code: ({ className, children, ...props }) => {
            const isInline = !className;
            if (isInline) {
              return (
                <code className="bg-muted px-2 py-1 rounded text-sm" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm mb-4">
              {children}
            </pre>
          ),
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
}
