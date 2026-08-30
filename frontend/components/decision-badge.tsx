import { Badge } from "@/components/ui/badge";
import type { Decision } from "@/data/types";

const VARIANT: Record<Decision, "go" | "review" | "hold" | "invalid"> = {
  GO: "go",
  REVIEW: "review",
  HOLD: "hold",
  INVALID: "invalid",
};

export function DecisionBadge({ decision }: { decision: Decision }) {
  return (
    <Badge variant={VARIANT[decision]} className="text-sm">
      {decision}
    </Badge>
  );
}
