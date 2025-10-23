"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { NavigationMenu, NavigationMenuItem, NavigationMenuList, navigationMenuTriggerStyle } from "@/components/ui/navigation-menu";
import { cn } from "@/lib/utils";

export function Header() {
  const pathname = usePathname();

  const isActive = (path: string) => {
    if (path === "/") {
      return pathname === "/";
    }
    return pathname.startsWith(path);
  };

  return (
    <header className="sticky top-0 z-50 w-full h-14 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto h-full flex items-center justify-between px-4">
        {/* Logo/Brand */}
        <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <Image
            src="/logo.png"
            alt="Homelab AI in Docker"
            width={32}
            height={32}
            className="h-8 w-8"
          />
          <span className="font-bold text-lg hidden sm:inline">Homelab AI in Docker</span>
          <span className="font-bold text-lg sm:hidden">HAID</span>
        </Link>

        {/* Navigation Menu */}
        <NavigationMenu className="hidden md:flex" viewport={false}>
          <NavigationMenuList>
            <NavigationMenuItem>
              <Link
                href="/"
                className={cn(
                  navigationMenuTriggerStyle(),
                  isActive("/") && "bg-accent text-accent-foreground"
                )}
              >
                Home
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link
                href="/models"
                className={cn(
                  navigationMenuTriggerStyle(),
                  isActive("/models") && "bg-accent text-accent-foreground"
                )}
              >
                Models
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link
                href="/status"
                className={cn(
                  navigationMenuTriggerStyle(),
                  isActive("/status") && "bg-accent text-accent-foreground"
                )}
              >
                Status
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link
                href="/settings"
                className={cn(
                  navigationMenuTriggerStyle(),
                  isActive("/settings") && "bg-accent text-accent-foreground"
                )}
              >
                Settings
              </Link>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>
      </div>
    </header>
  );
}
