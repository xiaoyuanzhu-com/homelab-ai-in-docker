"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown, ChevronUp } from "lucide-react";

export interface RawPayloadProps {
  label: string;
  payload: unknown;
  defaultOpen?: boolean;
  emptyMessage?: string;
}

export function RawPayload({
  label,
  payload,
  defaultOpen = false,
  emptyMessage = "No data available",
}: RawPayloadProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const content =
    payload === undefined || payload === null
      ? emptyMessage
      : typeof payload === "string"
      ? payload
      : JSON.stringify(payload, null, 2);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <Button variant="ghost" size="sm" className="w-full justify-between">
          <span className="text-sm font-medium">{label}</span>
          {isOpen ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-2">
        <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto max-h-96 whitespace-pre">
          {content}
        </pre>
      </CollapsibleContent>
    </Collapsible>
  );
}
