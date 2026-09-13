import { expect, test, type Page } from "@playwright/test";
import {
    createAuthenticatedApiContext,
    createAuthenticatedBrowserContext,
    createCodebook,
    createLabel,
    createProject,
    seedDemoCorpus,
    segmentCorpus,
    uniqueE2ECredentials,
    waitForRun,
    type E2ECredentials,
} from "./helpers/research-api";

/**
 * Phase 25 — Automated UI regression for the frontend redesign.
 * Prefer unique provisioned users; keep serial to protect shared API rate limits.
 */
test.describe.configure({ mode: "serial" });

const VIEWPORTS = [
    { name: "mobile", width: 390, height: 844 },
    { name: "tablet", width: 768, height: 1024 },
    { name: "desktop", width: 1440, height: 900 },
] as const;

function credentialsForTest(testInfo: {
    workerIndex: number;
    parallelIndex: number;
    retry: number;
    title: string;
}): E2ECredentials {
    const slug = testInfo.title
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "")
        .slice(0, 24);
    return uniqueE2ECredentials({
        prefix: `ux-${slug || "redesign"}`,
        workerIndex: testInfo.workerIndex,
        parallelIndex: testInfo.parallelIndex,
        retry: testInfo.retry,
    });
}

async function seedMinimalWorkspace(
    browser: import("@playwright/test").Browser,
    credentials: E2ECredentials,
    options?: { codebook?: boolean; segment?: boolean }
) {
    const api = await createAuthenticatedApiContext(browser, credentials, undefined, {
        provision: true,
    });
    const stamp = `${Date.now()}`;
    const project = await createProject(api, `UX Research ${stamp}`);
    const corpus = await seedDemoCorpus(api, project.id, `UX Demo ${stamp}`);
    let codebook: { id: string; name: string } | null = null;
    if (options?.codebook) {
        codebook = await createCodebook(api, project.id, `UX Codebook ${stamp}`);
        await createLabel(api, codebook.id, "Theme");
    }
    if (options?.segment) {
        const run = await segmentCorpus(api, corpus.id, "paragraph");
        if (run.status !== "completed") {
            await waitForRun(api, run.id);
        }
    }
    return { api, project, corpus, codebook };
}

async function openResearchPage(
    browser: import("@playwright/test").Browser,
    credentials: E2ECredentials,
    projectId: string,
    corpusId: string,
    path: string,
    codebookId?: string | null
) {
    const { context } = await createAuthenticatedBrowserContext(browser, credentials, {
        provision: false,
    });
    await context.addInitScript(
        ({ pid, cid, cbid }) => {
            localStorage.setItem(`text-research:corpus:${pid}`, cid);
            if (cbid) localStorage.setItem(`text-research:codebook:${pid}`, cbid);
        },
        { pid: projectId, cid: corpusId, cbid: codebookId ?? null }
    );
    const page = await context.newPage();
    await page.goto(path);
    return { context, page };
}

async function expectResearchShell(page: Page) {
    await expect(page.getByRole("navigation", { name: "Research navigation" })).toBeVisible({
        timeout: 30_000,
    });
    await expect(page.getByRole("region", { name: "Research context" })).toBeVisible();
}

test.describe("Phase 25 UX redesign regression", () => {
    test("app chrome: skip link, main landmark, and projects navigation", async ({
        browser,
    }, testInfo) => {
        test.setTimeout(90_000);
        const credentials = credentialsForTest(testInfo);
        const api = await createAuthenticatedApiContext(browser, credentials);
        try {
            await createProject(api, `UX Nav ${Date.now()}`);
            const { context } = await createAuthenticatedBrowserContext(browser, credentials, {
                provision: false,
            });
            const page = await context.newPage();
            await page.goto("/projects");
            await expect(page.getByRole("main")).toBeVisible({ timeout: 20_000 });
            await expect(page.getByRole("link", { name: "Skip to main content" })).toBeAttached();
            await expect(page.getByRole("button", { name: /Open navigation menu|Collapse menu|Expand menu/i }).or(
                page.getByRole("link", { name: /Projects/i })
            ).first()).toBeVisible();
            await page.getByRole("link", { name: /Projects/i }).first().click();
            await expect(page).toHaveURL(/\/projects/);
            await context.close();
        } finally {
            await api.close();
        }
    });

    test("grouped research navigation and page tabs", async ({ browser }, testInfo) => {
        test.setTimeout(120_000);
        const credentials = credentialsForTest(testInfo);
        const { api, project, corpus, codebook } = await seedMinimalWorkspace(browser, credentials, {
            codebook: true,
        });
        try {
            const { context, page } = await openResearchPage(
                browser,
                credentials,
                project.id,
                corpus.id,
                `/research/${project.id}/dashboard`,
                codebook?.id
            );
            await expectResearchShell(page);

            const nav = page.getByRole("navigation", { name: "Research navigation" });
            for (const group of ["Overview", "Data", "Coding & Quality", "Analysis", "Models", "Outputs"]) {
                await expect(nav.getByRole("button", { name: group, exact: true })).toBeVisible();
            }

            await nav.getByRole("button", { name: "Data", exact: true }).click();
            await nav.getByRole("button", { name: "Corpus", exact: true }).or(
                page.getByRole("button", { name: "Corpus", exact: true })
            ).first().click();
            await expect(page).toHaveURL(new RegExp(`/research/${project.id}/corpus`));

            await page.goto(`/research/${project.id}/prepare?tab=segment`);
            await expect(page.getByRole("tab", { name: /Segment/i })).toBeVisible({
                timeout: 20_000,
            });
            await page.getByRole("tab", { name: /Cleaning/i }).click();
            await expect(page).toHaveURL(/tab=cleaning/);

            await page.goto(`/research/${project.id}/classification?tab=setup`);
            await expect(page.getByRole("tab", { name: /Setup|Train|Dataset/i }).first()).toBeVisible({
                timeout: 20_000,
            });
            await page.getByRole("tab", { name: /Train/i }).click();
            await expect(page).toHaveURL(/tab=train/);

            await context.close();
        } finally {
            await api.close();
        }
    });

    test("context bar change confirmation dialog", async ({ browser }, testInfo) => {
        test.setTimeout(120_000);
        const credentials = credentialsForTest(testInfo);
        const api = await createAuthenticatedApiContext(browser, credentials);
        try {
            const stamp = `${Date.now()}`;
            const project = await createProject(api, `UX Context ${stamp}`);
            const corpusA = await seedDemoCorpus(api, project.id, `UX Corpus A ${stamp}`);
            const corpusB = await seedDemoCorpus(api, project.id, `UX Corpus B ${stamp}`);
            const { context, page } = await openResearchPage(
                browser,
                credentials,
                project.id,
                corpusA.id,
                `/research/${project.id}/dashboard`
            );
            await expectResearchShell(page);

            const contextRegion = page.getByRole("region", { name: "Research context" });
            await contextRegion.getByLabel("Corpus").click();
            await page.getByRole("option", { name: corpusB.name }).click();
            await expect(page.getByRole("dialog", { name: /Change research context/i })).toBeVisible();
            await page.getByRole("button", { name: "Change", exact: true }).click();
            await expect(page.getByRole("dialog")).toHaveCount(0);
            await expect(contextRegion.getByText(corpusB.name).or(contextRegion.getByLabel("Corpus"))).toBeVisible();

            await context.close();
        } finally {
            await api.close();
        }
    });

    test("empty state and help tooltip accessibility", async ({ browser }, testInfo) => {
        test.setTimeout(120_000);
        const credentials = credentialsForTest(testInfo);
        const { api, project, corpus, codebook } = await seedMinimalWorkspace(browser, credentials, {
            codebook: true,
        });
        try {
            const { context } = await createAuthenticatedBrowserContext(browser, credentials, {
                provision: false,
            });
            // Intentionally omit corpus localStorage to exercise empty / prompt states.
            const page = await context.newPage();
            await page.goto(`/research/${project.id}/dashboard`);
            await expect(
                page.getByRole("heading", { name: /No corpus yet/i }).or(
                    page.getByText(/Select a corpus|No corpus|Create codebook|Documents/i)
                ).first()
            ).toBeVisible({ timeout: 30_000 });

            await context.addInitScript(
                ({ pid, cid, cbid }) => {
                    localStorage.setItem(`text-research:corpus:${pid}`, cid);
                    localStorage.setItem(`text-research:codebook:${pid}`, cbid);
                },
                { pid: project.id, cid: corpus.id, cbid: codebook!.id }
            );
            await page.goto(`/research/${project.id}/reliability?tab=overview`);
            await expect(page.getByRole("navigation", { name: "Research navigation" })).toBeVisible({
                timeout: 30_000,
            });
            const help = page.getByRole("button", { name: /About /i }).first();
            if (await help.isVisible().catch(() => false)) {
                await help.focus();
                await expect(help).toBeFocused();
                await help.click();
                await expect(
                    page.getByRole("dialog").or(page.getByRole("tooltip")).first()
                ).toBeVisible({ timeout: 5_000 });
            }

            await context.close();
        } finally {
            await api.close();
        }
    });

    test("long-running job status panel on Prepare", async ({ browser }, testInfo) => {
        test.setTimeout(120_000);
        const credentials = credentialsForTest(testInfo);
        const { api, project, corpus } = await seedMinimalWorkspace(browser, credentials, {
            segment: true,
        });
        try {
            const { context, page } = await openResearchPage(
                browser,
                credentials,
                project.id,
                corpus.id,
                `/research/${project.id}/prepare?tab=segment`
            );
            await expect(page.getByText(/Run status|Segmentation|Completed|Queued|Running/i).first()).toBeVisible({
                timeout: 30_000,
            });
            await expect(
                page.getByLabel(/Status:/i).or(page.getByText(/Completed|Running|Queued/i)).first()
            ).toBeVisible();
            await context.close();
        } finally {
            await api.close();
        }
    });

    test("annotation, classification, and RAG workspace smoke", async ({ browser }, testInfo) => {
        test.setTimeout(120_000);
        const credentials = credentialsForTest(testInfo);
        const { api, project, corpus, codebook } = await seedMinimalWorkspace(browser, credentials, {
            codebook: true,
            segment: true,
        });
        try {
            const { context, page } = await openResearchPage(
                browser,
                credentials,
                project.id,
                corpus.id,
                `/research/${project.id}/annotation`,
                codebook?.id
            );
            await expect(page.getByRole("heading", { name: /Annotation|Coding/i }).or(
                page.getByText(/annotat|codebook|unit/i)
            ).first()).toBeVisible({ timeout: 30_000 });

            await page.goto(`/research/${project.id}/classification?tab=dataset`);
            await expect(page.getByRole("tab", { name: /Dataset|Train|Setup/i }).first()).toBeVisible({
                timeout: 20_000,
            });
            await expect(page.getByText(/dataset|snapshot|Freeze|Train/i).first()).toBeVisible();

            await page.goto("/rag");
            await expect(
                page.getByRole("heading", { name: /RAG|Retrieval/i }).or(page.getByRole("tab", { name: /Documents/i }))
            ).first().toBeVisible({ timeout: 30_000 });
            await expect(page.getByRole("tab", { name: /Documents|Indexing|Retrieval/i }).first()).toBeVisible();

            await context.close();
        } finally {
            await api.close();
        }
    });

    for (const viewport of VIEWPORTS) {
        test(`responsive research shell @ ${viewport.name} (${viewport.width}px)`, async ({
            browser,
        }, testInfo) => {
            test.setTimeout(120_000);
            const credentials = credentialsForTest(testInfo);
            const { api, project, corpus } = await seedMinimalWorkspace(browser, credentials);
            try {
                const { context, page } = await openResearchPage(
                    browser,
                    credentials,
                    project.id,
                    corpus.id,
                    `/research/${project.id}/dashboard`
                );
                await page.setViewportSize({ width: viewport.width, height: viewport.height });
                await expectResearchShell(page);

                if (viewport.width < 900) {
                    await expect(
                        page.getByRole("button", { name: /Open research navigation|Navigate/i })
                    ).toBeVisible();
                    await page.getByRole("button", { name: /Open research navigation|Navigate/i }).click();
                    await expect(page.getByText(/Overview|Data|Progress/i).first()).toBeVisible();
                } else {
                    await expect(
                        page.getByRole("navigation", { name: "Research navigation" }).getByRole("button", {
                            name: "Overview",
                            exact: true,
                        })
                    ).toBeVisible();
                }

                // Ask Corpus: drawer on narrow/mid, inline from lg (1200+).
                if (viewport.width < 1200) {
                    const ask = page.getByRole("button", { name: "Ask Corpus", exact: true });
                    if (await ask.isVisible().catch(() => false)) {
                        await ask.click();
                        await expect(page.getByRole("tab", { name: "Ask" }).or(
                            page.getByText("Ask Corpus")
                        ).first()).toBeVisible({ timeout: 10_000 });
                    }
                }

                if (process.env.E2E_VISUAL === "1") {
                    await expect(page.getByRole("main")).toHaveScreenshot(
                        `research-shell-${viewport.name}.png`,
                        { maxDiffPixelRatio: 0.08 }
                    );
                }

                await context.close();
            } finally {
                await api.close();
            }
        });
    }
});
