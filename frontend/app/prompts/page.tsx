import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { DecisionBadge } from "@/components/decision-badge";
import { PROMPT_STORY } from "@/data/experiments";
import { getPromptTemplate } from "@/data/prompts";

export default function PromptsPage() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Prompts</h1>
        <p className="mt-2 max-w-2xl text-ink-muted">
          The exact template text for each version, verbatim from{" "}
          <code className="font-mono text-xs">prompts/</code> — nothing paraphrased.
        </p>
      </div>

      <Tabs defaultValue="v1">
        <TabsList>
          {PROMPT_STORY.map((p) => (
            <TabsTrigger key={p.version} value={p.version}>
              {p.title}
            </TabsTrigger>
          ))}
        </TabsList>

        {PROMPT_STORY.map((p) => (
          <TabsContent key={p.version} value={p.version}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>{p.title}</CardTitle>
                  {p.decision && <DecisionBadge decision={p.decision} />}
                </div>
                <CardDescription>{p.summary}</CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="overflow-x-auto rounded-md bg-surface-muted p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap text-ink">
                  {getPromptTemplate(p.version)}
                </pre>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>

      <Card>
        <CardHeader>
          <CardTitle>What changed, version to version</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-sm text-ink-muted">
          <p>
            <strong className="text-ink">V1 → V2:</strong> replaced a plain refusal instruction
            with an emphatic, ordered insufficient-information rule, to fix V1&apos;s severe
            under-abstention (6.67% abstention accuracy). This worked — but caused a real,
            CI-confirmed faithfulness regression (see{" "}
            <a href="/failure-analysis" className="underline underline-offset-2">
              Failure Analysis
            </a>
            ).
          </p>
          <p>
            <strong className="text-ink">V2 → V3:</strong> added a check-before-refusing step
            (rule 2) and required restating specifically what information is missing (rule 3),
            targeting the two root causes behind V2&apos;s regression without weakening the
            insufficient-information rule that fixed abstention.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
