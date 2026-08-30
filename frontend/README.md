# GenAI ReleaseGate — frontend

A Next.js (App Router, TypeScript, Tailwind CSS v4, shadcn/ui-style components, Recharts)
visualization layer over the real results produced by the Python evaluation pipeline in
this repository. It renders **only** already-computed, already-committed data — it never
calls Bedrock, never runs an evaluation, and needs no API keys or environment variables.

## Data flow

```
../results/reports/*.json  ──┐
../prompts/*.md             ─┼──►  data/*.ts (server-only fs reads)  ──►  app/*/page.tsx
../scripts/compare_guardrail_versions.py's cited output (data/safety.ts)  ─┘
```

`data/reports.ts` and `data/prompts.ts` are the only two files that touch the filesystem
(both marked `"server-only"`). Every page is a Server Component that imports from `data/`
— nothing is fetched client-side, and nothing is hand-typed into a component.

## Local development

```bash
cd frontend
npm install
npm run dev      # http://localhost:3000
```

## Build

```bash
npm run build    # static-generates all 8 routes
npm run start    # serve the production build locally
```

## Deploying to Vercel

1. Import this GitHub repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Framework preset: Next.js (auto-detected).
4. Build command / output: defaults (`next build`).
5. **No environment variables are required** — the app reads `../results/reports/*.json`
   and `../prompts/*.md` directly from the repo checkout at build time; Vercel clones the
   whole repository, so those paths resolve correctly with Root Directory set to
   `frontend`.
6. To refresh the site with new results, regenerate the JSON reports in the Python
   backend (`uv run python scripts/generate_reports.py`), commit them, and redeploy —
   the frontend never computes new evaluation results itself.
