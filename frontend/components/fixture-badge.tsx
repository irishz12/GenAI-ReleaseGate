import { Badge } from "@/components/ui/badge";

/** The one visual element every synthetic decision-engine example must
 * carry, at the same size/weight as a real decision badge next to it —
 * never smaller, never relegated to prose alone. */
export function FixtureBadge() {
  return <Badge variant="fixture">VALIDATION FIXTURE</Badge>;
}
