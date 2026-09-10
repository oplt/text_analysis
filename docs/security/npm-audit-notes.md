# Frontend npm audit notes (Phase 20)

Reviewed: 2026-09-10.

## Commands

```bash
cd frontend
npm audit
npm audit --omit=dev
```

## Classification (full audit)

| Package | Severity | Class | Reachable in production UI? | Notes |
|---------|----------|-------|-----------------------------|-------|
| `vite` | high | **dev-only** (build tooling) | No | Dev server / bundler. Not shipped to browsers. |
| `esbuild` | moderate | **dev-only** transitive (via vite) | No | Local transform server advisory. |
| `vitest` | moderate | **dev-only** | No | Unit test runner. |
| `@vitest/coverage-v8` | moderate | **dev-only** | No | Coverage reporter. |
| `@vitest/mocker` | moderate | **dev-only** transitive | No | Pulled by vitest. |

`npm audit --omit=dev` reported **0** production runtime vulnerabilities at review time.

## Policy

- Do **not** run `npm audit fix --force` without reviewing breaking major bumps.
- Prefer bumping Vite/Vitest on a dedicated dependency PR when convenient.
- Runtime (browser) dependency surface is currently clean of reported advisories.
