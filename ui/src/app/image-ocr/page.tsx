"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function ImageOCRPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Image OCR</h1>
        <p className="text-muted-foreground">
          Extract text from images using optical character recognition
        </p>
      </div>

      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Coming Soon</AlertTitle>
        <AlertDescription>
          Image OCR functionality is currently under development. PaddleOCR integration is planned
          for a future release.
        </AlertDescription>
      </Alert>

      <div className="mt-6">
        <Card>
          <CardHeader>
            <CardTitle>Planned Features</CardTitle>
            <CardDescription>What to expect when OCR is implemented</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>Extract text from images with high accuracy</li>
              <li>Support for multiple languages</li>
              <li>Handle various image formats (PNG, JPG, PDF)</li>
              <li>Detect and preserve text layout</li>
              <li>Batch processing capabilities</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
