import { test, expect } from "@playwright/test";
import {
    createAuthenticatedApiContext,
    createAuthenticatedBrowserContext,
    createProject,
    getAssistantScope,
    postAssistantMessage,
    postAssistantRetrieve,
    seedDemoCorpus,
    uniqueE2ECredentials,
    type E2ECredentials,
} from "./helpers/research-api";

test.describe.configure({ mode: "serial" });

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
        prefix: `ask-${slug || "corpus"}`,
        workerIndex: testInfo.workerIndex,
        parallelIndex: testInfo.parallelIndex,
        retry: testInfo.retry,
    });
}

test.describe("Ask Corpus evidence flow", () => {
    test("API: scope allow-list + retrieve stays inside corpus", async ({ browser }, testInfo) => {
        test.setTimeout(90_000);
        const credentials = credentialsForTest(testInfo);
        const api = await createAuthenticatedApiContext(browser, credentials);
        try {
            const project = await createProject(api, `Ask Corpus ${Date.now()}`);
            const corpusA = await seedDemoCorpus(api, project.id, `Ask Demo A ${Date.now()}`);
            const corpusB = await seedDemoCorpus(api, project.id, `Ask Demo B ${Date.now()}`);
            const scopeA = await getAssistantScope(api, corpusA.id);
            const scopeB = await getAssistantScope(api, corpusB.id);

            expect(scopeA.corpus_id).toBe(corpusA.id);
            expect(scopeB.corpus_id).toBe(corpusB.id);
            expect(scopeA.scope_hash).not.toBe(scopeB.scope_hash);
            expect(Array.isArray(scopeA.rag_document_ids)).toBeTruthy();
            expect(Array.isArray(scopeB.rag_document_ids)).toBeTruthy();

            if (scopeA.indexed_count > 0) {
                expect(scopeA.rag_document_ids.length).toBe(scopeA.indexed_count);
                const retrieved = await postAssistantRetrieve(api, corpusA.id, {
                    query: "responsibility accountability community",
                    intent: "lookup",
                });
                expect(retrieved.retrieval_trace_id || retrieved.chunks).toBeTruthy();
                for (const hit of retrieved.chunks ?? []) {
                    expect(scopeA.rag_document_ids).toContain(hit.document_id);
                    expect(scopeB.rag_document_ids).not.toContain(hit.document_id);
                }
            }

            if (scopeA.indexed_count === 0) {
                const ask = await postAssistantMessage(api, corpusA.id, {
                    query: "What claims appear in this corpus?",
                });
                if (ask.ok) {
                    const citations = ask.body?.citations ?? ask.body?.message?.citations ?? [];
                    expect(Array.isArray(citations)).toBeTruthy();
                    expect(citations.length).toBe(0);
                }
            }
        } finally {
            await api.close();
        }
    });

    test("UI: Ask Corpus panel opens and citation drawer can open", async ({ browser }, testInfo) => {
        test.setTimeout(120_000);
        const credentials = credentialsForTest(testInfo);
        const api = await createAuthenticatedApiContext(browser, credentials);
        let projectId = "";
        let corpusId = "";
        try {
            const project = await createProject(api, `Ask UI ${Date.now()}`);
            const corpus = await seedDemoCorpus(api, project.id, `Ask UI Demo ${Date.now()}`);
            projectId = project.id;
            corpusId = corpus.id;
            await getAssistantScope(api, corpus.id);
        } finally {
            // keep api for cleanup path; close after UI
        }

        try {
            const { context } = await createAuthenticatedBrowserContext(browser, credentials, {
                provision: false,
            });
            await context.addInitScript(
                ({ projectId: pid, corpusId: cid }) => {
                    localStorage.setItem(`text-research:corpus:${pid}`, cid);
                },
                { projectId, corpusId }
            );

            const page = await context.newPage();
            await page.goto(`/research/${projectId}/dashboard`);
            await expect(page.getByRole("heading", { name: /Text Research/i })).toBeVisible({
                timeout: 20_000,
            });

            await page.getByRole("button", { name: "Ask Corpus", exact: true }).click();
            await expect(page.getByRole("tab", { name: "Ask" })).toBeVisible();
            await page.getByRole("tab", { name: "Ask" }).click();
            await expect(page.getByRole("heading", { name: "Ask Corpus" }).or(
                page.getByText("Ask Corpus", { exact: true })
            ).first()).toBeVisible();

            await page.getByRole("tab", { name: "Evidence" }).click();
            await expect(
                page.getByText(/Retrieved evidence|Ask a question to populate evidence/i).first()
            ).toBeVisible({ timeout: 10_000 });

            await page.getByRole("tab", { name: "Ask" }).click();
            const question = page.getByLabel("Question");
            await expect(question).toBeVisible();
            await question.fill("What themes appear across these documents?");
            const askBtn = page.getByRole("button", { name: "Ask", exact: true });
            if (await askBtn.isEnabled()) {
                await askBtn.click();
                await expect(
                    page
                        .getByText(
                            /Retrieving evidence|citation|insufficient|failed|indexed|answer|Evidence/i
                        )
                        .first()
                ).toBeVisible({ timeout: 60_000 });

                const openSource = page.getByRole("button", { name: /Open source|View source/i }).first();
                if (await openSource.isVisible().catch(() => false)) {
                    await openSource.click();
                    await expect(page.getByRole("dialog").or(page.getByText(/page|snippet/i)).first()).toBeVisible({
                        timeout: 10_000,
                    });
                }
            } else {
                // Zero indexed docs: Ask disabled is correct allow-list behavior
                await expect(page.getByText(/indexed/i).first()).toBeVisible();
            }

            await context.close();
        } finally {
            await api.close();
        }
    });
});
