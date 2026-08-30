"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";
import { ThemeToggle } from "./theme-toggle";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/prompts", label: "Prompts" },
  { href: "/safety", label: "Safety" },
  { href: "/statistics", label: "Statistics" },
  { href: "/release-gate", label: "Release Gate" },
  { href: "/failure-analysis", label: "Failure Analysis" },
  { href: "/methodology", label: "Methodology" },
];

export function SiteNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-rule bg-bg/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <Link href="/" className="shrink-0 font-mono text-sm font-semibold tracking-tight">
          GenAI <span className="text-accent">ReleaseGate</span>
        </Link>
        <nav className="scrollbar-none flex flex-1 items-center gap-1 overflow-x-auto">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "shrink-0 rounded-md px-3 py-1.5 text-sm font-medium whitespace-nowrap transition-colors",
                  active
                    ? "bg-surface-muted text-ink"
                    : "text-ink-muted hover:bg-surface-muted hover:text-ink",
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
        <ThemeToggle />
      </div>
    </header>
  );
}
