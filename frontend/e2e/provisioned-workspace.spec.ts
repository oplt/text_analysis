import { expect, test, type Browser } from "@playwright/test";

const email = process.env.E2E_TEST_EMAIL;
const password = process.env.E2E_TEST_PASSWORD;
const adminEmail = process.env.E2E_ADMIN_EMAIL;
const adminPassword = process.env.E2E_ADMIN_PASSWORD;
const apiBaseUrl = process.env.E2E_API_URL ?? "http://localhost:8000";

async function authenticatedPage(browser: Browser, userEmail: string, userPassword: string) {
    const context = await browser.newContext();
    const signIn = await context.request.post(`${apiBaseUrl}/api/v1/auth/sign-in`, {
        data: { email: userEmail, password: userPassword },
    });
    expect(signIn.ok()).toBeTruthy();
    return { context, page: await context.newPage() };
}

test.describe("provisioned user workspace", () => {
    test.skip(!email || !password, "Set E2E_TEST_EMAIL and E2E_TEST_PASSWORD");

    for (const [name, path, heading] of [
        ["platform", "/platform", /platform/i],
        ["profile", "/profile", /profile/i],
        ["projects", "/projects", /projects/i],
        ["notifications", "/notifications", /notifications/i],
    ] as const) {
        test(`${name} flow loads authenticated data`, async ({ browser }) => {
            const { context, page } = await authenticatedPage(browser, email!, password!);
            await page.goto(path);
            await expect(page.getByRole("heading", { name: heading }).first()).toBeVisible();
            await expect(page).toHaveURL(new RegExp(`${path.replace("/", "\\/")}$`));
            await context.close();
        });
    }
});

test.describe("provisioned admin workspace", () => {
    test.skip(!adminEmail || !adminPassword, "Set E2E_ADMIN_EMAIL and E2E_ADMIN_PASSWORD");

    for (const [name, path] of [
        ["settings", "/admin/settings"],
        ["platform", "/admin/platform"],
        ["users", "/admin/users"],
    ] as const) {
        test(`admin ${name} flow passes authorization`, async ({ browser }) => {
            const { context, page } = await authenticatedPage(browser, adminEmail!, adminPassword!);
            await page.goto(path);
            await expect(page).toHaveURL(new RegExp(`${path.replaceAll("/", "\\/")}$`));
            await expect(page.locator("main")).toBeVisible();
            await context.close();
        });
    }
});
