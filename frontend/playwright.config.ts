import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://localhost:5173";

export default defineConfig({
    testDir: "./e2e",
    fullyParallel: true,
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
