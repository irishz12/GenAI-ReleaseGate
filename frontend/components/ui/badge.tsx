import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold whitespace-nowrap",
  {
    variants: {
      variant: {
        default: "border-rule bg-surface-muted text-ink",
        go: "border-go-rule bg-go-bg text-go",
        review: "border-review-rule bg-review-bg text-review",
        hold: "border-hold-rule bg-hold-bg text-hold",
        invalid: "border-invalid-rule bg-invalid-bg text-invalid",
        // Distinct from every decision-status color on purpose: this must
        // never be mistaken, at a skim or in a screenshot, for a real GO.
        fixture: "border-dashed border-accent bg-transparent text-accent",
        real: "border-ink bg-ink text-bg",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

function Badge({
  className,
  variant,
  ...props
}: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
