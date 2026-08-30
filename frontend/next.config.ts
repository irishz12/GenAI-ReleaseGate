import type { NextConfig } from "next";

// This app renders exclusively from files already committed to the repo
// (results/reports/*.json) via Node's fs at build/render time — there is no
// API route, no rewrite, no external image domain, and no reason to relax
// any default. Kept intentionally minimal.
const nextConfig: NextConfig = {};

export default nextConfig;
