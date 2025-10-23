"use client";

import { ReactNode } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

export interface SkillOption {
  value: string;
  label: string;
  description?: string;
}

interface SkillSelectorProps {
  value: string;
  onChange?: (value: string) => void;
  options: SkillOption[];
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  emptyMessage?: string;
  emptyContent?: ReactNode;
}

export function SkillSelector({
  value,
  onChange,
  options,
  loading,
  disabled,
  placeholder = "Select a skill",
  emptyMessage = "No skills available. Download a skill first.",
  emptyContent,
}: SkillSelectorProps) {
  if (loading) {
    return <Skeleton className="h-10 w-full" />;
  }

  if (options.length === 0) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          {emptyContent ?? emptyMessage}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
