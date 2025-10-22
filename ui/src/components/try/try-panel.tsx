"use client";

import { ReactNode } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { RawPayload, RawPayloadProps } from "./raw-payload";

export interface TryPanelProps {
  title: string;
  description?: string;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
  rawPayload?: RawPayloadProps;
}

export function TryPanel({
  title,
  description,
  children,
  footer,
  className,
  rawPayload,
}: TryPanelProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description ? <CardDescription>{description}</CardDescription> : null}
      </CardHeader>
      <CardContent className="space-y-4">
        {children}
        {footer}
        {rawPayload ? <RawPayload {...rawPayload} /> : null}
      </CardContent>
    </Card>
  );
}
