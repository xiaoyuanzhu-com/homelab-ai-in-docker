"use client"

import * as React from "react"
import { Moon, Sun, SunMoon } from "lucide-react"
import { useTheme } from "next-themes"

export function ThemeSwitcher() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  // Avoid hydration mismatch by only rendering after mount
  React.useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <div className="w-[14px] h-[14px]" aria-hidden="true" />
    )
  }

  const cycleTheme = () => {
    if (theme === "light") {
      setTheme("dark")
    } else if (theme === "dark") {
      setTheme("system")
    } else {
      setTheme("light")
    }
  }

  const getIcon = () => {
    switch (theme) {
      case "light":
        return <Sun className="w-[14px] h-[14px]" />
      case "dark":
        return <Moon className="w-[14px] h-[14px]" />
      default:
        return <SunMoon className="w-[14px] h-[14px]" />
    }
  }

  const getTitle = () => {
    switch (theme) {
      case "light":
        return "Switch to dark mode"
      case "dark":
        return "Switch to system mode"
      default:
        return "Switch to light mode"
    }
  }

  return (
    <button
      onClick={cycleTheme}
      className="text-muted-foreground hover:text-foreground transition-colors inline-flex"
      title={getTitle()}
      aria-label={getTitle()}
    >
      {getIcon()}
    </button>
  )
}
