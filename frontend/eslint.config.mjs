import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

// Both eslint-config-next exports are already flat-config arrays (built
// directly from @next/eslint-plugin-next's own flat configs) — no
// @eslint/eslintrc / FlatCompat legacy bridge needed for Next.js 16.
const eslintConfig = [...nextCoreWebVitals, ...nextTypescript];

export default eslintConfig;
