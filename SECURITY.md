# Security Policy

## Supported versions

Security fixes are provided for the actively maintained `main` branch.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report security issues privately to the project maintainers through your organization's preferred secure channel. If no channel is listed in the repository settings, contact the repository owner directly.

Include:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested remediation, if known

We will acknowledge receipt and work with you on a coordinated disclosure timeline.

## Frontend dependency audit (Phase 0 baseline)

Last reviewed: 2026-09-09.

### Actions taken (safe upgrades, no `--force`)

| Package | Change | Exposure |
|---------|--------|----------|
| `axios` | → `^1.20.0` | **Runtime** (browser HTTP client) |
| `react-router-dom` / `react-router` | → `^7.18.3` | **Runtime** (SPA routing) |
| `vitest` / `@vitest/coverage-v8` | → `^3.2.7` | **Dev-only** (unit tests; clears prior critical Vitest UI advisory on `<3.2.6`) |
| Transitive fixes via `npm audit fix` | applied where non-breaking | Mixed |

### Remaining exceptions (intentionally deferred)

Do **not** run `npm audit fix --force` for these: it would jump to Vite 8 / Vitest 5 and risk breaking the current toolchain.

| Advisory area | Severity | Scope | Why deferred | Mitigation |
|---------------|----------|-------|--------------|------------|
| `vitest` / `@vitest/mocker` path traversal (needs Vitest ≥4.1.11 / 5.x) | moderate | **Dev-only** | Major vitest upgrade | Do not expose Vitest UI / mocker redirect mocks to untrusted networks; CI uses `vitest run` locally |
| `vite` ≤6.4.x via `esbuild` dev-server request reflection | moderate/high (tooling) | **Dev-only** (`npm run dev`) | Vite 8 major bump | Bind Vite to localhost; do not expose the Vite dev server publicly |
| Nested Vite 5 `esbuild` advisory | moderate | **Dev-only** | Same as above | Same as above |

Production builds (`npm run build` → static assets served without Vite/esbuild/vitest) are not affected by the deferred **dev-server** advisories. Runtime client deps (`axios`, `react-router-dom`) were patched above.
