import { test, expect } from "@playwright/test";

const email = process.env.E2E_TEST_EMAIL;
const password = process.env.E2E_TEST_PASSWORD;
const apiBaseUrl = process.env.E2E_API_URL ?? "http://localhost:8000";

async function createAuthenticatedContext(browser: import("@playwright/test").Browser) {
    const context = await browser.newContext();
    const signIn = await context.request.post(`${apiBaseUrl}/api/v1/auth/sign-in`, {
        data: { email, password },
    });
    expect(signIn.ok()).toBeTruthy();
    return context;
}

test.describe("workspace smoke", () => {
    test("redirects signed-out users away from dashboard", async ({ page }) => {
        await page.goto("/dashboard");
        await expect(page).toHaveURL(/\/$/);
    });

    test("login page renders sign-in controls", async ({ page }) => {
        await page.goto("/");
        await expect(page.getByRole("button", { name: /sign in/i })).toBeVisible();
    });
});

test.describe("authenticated AI flow", () => {
    test.skip(!email || !password, "Set E2E_TEST_EMAIL and E2E_TEST_PASSWORD");

    test("ingest a text document in AI Studio", async ({ browser }) => {
        const context = await createAuthenticatedContext(browser);
        const page = await context.newPage();

        await page.goto("/ai");
        await expect(page.getByRole("heading", { name: /AI Studio/i })).toBeVisible();

        await page.getByLabel("Document title").fill("E2E Notes");
        await page.getByLabel("Document content").fill(
            "Playwright uploaded this document for retrieval tests."
        );
        await page.getByRole("button", { name: "Create text document" }).click();

        await expect(page.getByText("Document ingested.")).toBeVisible({ timeout: 15_000 });

        await context.close();
    });

    test("ask via agent run API after document ingest", async ({ browser }) => {
        const context = await createAuthenticatedContext(browser);
        const csrfToken = (await context.cookies()).find((cookie) => cookie.name === "csrf_token")?.value;

        const templateKey = `e2e_${Date.now()}`;
        const template = await context.request.post(`${apiBaseUrl}/api/v1/ai/prompts`, {
            headers: csrfToken ? { "X-CSRF-Token": csrfToken } : undefined,
            data: {
                key: templateKey,
                name: "E2E Prompt",
                description: "Prompt for Playwright smoke test",
            },
        });
        expect(template.ok()).toBeTruthy();
        const templateId = (await template.json()).id;

        const version = await context.request.post(
            `${apiBaseUrl}/api/v1/ai/prompts/${templateId}/versions`,
            {
                headers: csrfToken ? { "X-CSRF-Token": csrfToken } : undefined,
                data: {
                    provider_key: "local",
                    model_name: "local-heuristic",
                    system_prompt: "You are concise.",
                    user_prompt_template: "Answer: {{question}}",
                    variable_definitions: [{ name: "question", required: true }],
                    is_published: true,
                },
            }
        );
        expect(version.ok()).toBeTruthy();

        const run = await context.request.post(`${apiBaseUrl}/api/v1/agent/runs`, {
            headers: csrfToken ? { "X-CSRF-Token": csrfToken } : undefined,
            data: {
                prompt_template_key: templateKey,
                variables: { question: "Say hello from e2e." },
                user_message: "Say hello from e2e.",
            },
        });
        expect(run.ok()).toBeTruthy();
        const body = await run.json();
        expect(body.status).toBe("completed");
        expect(body.output_text).toBeTruthy();

        await context.close();
    });
});
