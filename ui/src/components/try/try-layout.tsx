"use client";

import { cn } from "@/lib/utils";
import { TryPanel, TryPanelProps } from "./try-panel";

export interface TryLayoutProps {
  input: TryPanelProps;
  output: TryPanelProps;
  className?: string;
}

export function TryLayout({ input, output, className }: TryLayoutProps) {
  return (
    <div
      className={cn(
        "grid gap-6 grid-cols-1 lg:grid-cols-[minmax(0,1fr),minmax(0,1fr)]",
        className,
      )}
    >
      <TryPanel {...input} />
      <TryPanel {...output} />
    </div>
  );
}
