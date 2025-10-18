"use client";

import Link from "next/link";
import { NavigationMenu, NavigationMenuItem, NavigationMenuList, navigationMenuTriggerStyle } from "@/components/ui/navigation-menu";

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full h-14 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto h-full flex items-center justify-between px-4">
        {/* Logo/Brand */}
        <Link href="/" className="flex items-center gap-2">
          <div className="text-2xl">🤖</div>
          <span className="font-bold text-lg">Homelab AI</span>
        </Link>

        {/* Navigation Menu */}
        <NavigationMenu className="hidden md:flex" viewport={false}>
          <NavigationMenuList>
            <NavigationMenuItem>
              <Link href="/" className={navigationMenuTriggerStyle()}>
                Home
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link href="/embedding" className={navigationMenuTriggerStyle()}>
                Embedding
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link href="/image-caption" className={navigationMenuTriggerStyle()}>
                Image Caption
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link href="/crawl" className={navigationMenuTriggerStyle()}>
                Crawl
              </Link>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>
      </div>
    </header>
  );
}
