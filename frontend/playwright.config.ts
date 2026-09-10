import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://localhost:5173";

/**
 * Research E2E hits a shared API IP with many mutating requests.
 * Unique users isolate auth rate-limit keys; serial workers avoid public IP 429s
 * under CI retries. Override with E2E_WORKERS when intentionally parallelizing.
 */
const configuredWorkers = process.env.E2E_WORKERS
    ? Number.parseInt(process.env.E2E_WORKERS, 10)
    : process.env.CI
      ? 1
      : undefined;

export default defineConfig({
    testDir: "./e2e",
    fullyParallel: configuredWorkers === 1 ? false : true,
    workers: configuredWorkers,
    retries: process.env.CI ? 2 : 0,
    reporter: process.env.CI ? "github" : "html",
    timeout: 60_000,
    use: {
        baseURL,
        trace: "on-first-retry",
    },
    projects: process.env.CI
        ? [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }]
        : [
              { name: "chromium", use: { ...devices["Desktop Chrome"] } },
              { name: "firefox", use: { ...devices["Desktop Firefox"] } },
              { name: "webkit", use: { ...devices["Desktop Safari"] } },
          ],
    webServer: process.env.E2E_BASE_URL
        ? undefined
        : {
              command: "npm run dev",
              url: baseURL,
              reuseExistingServer: !process.env.CI,
          },
});
