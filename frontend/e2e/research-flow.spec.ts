import { test, expect } from "@playwright/test";
import {
    computeReliability,
    createAuthenticatedApiContext,
    createAuthenticatedBrowserContext,
    createCodebook,
    createLabel,
    createProject,
    freezeDataset,
    getDashboard,
    listClassifiers,
    listPredictions,
    listTextUnitIdsFromExport,
    predictWithModel,
    previewDataset,
    saveAnnotations,
    segmentCorpus,
    seedDemoCorpus,
    trainClassifier,
    uniqueE2ECredentials,
    waitForRun,
    type ApiContext,
    type E2ECredentials,
} from "./helpers/research-api";

/**
 * Research E2E is stateful and chatty against a shared API IP.
 * Prefer unique users per test; run this file serially (also workers:1 in CI).
 */
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
        prefix: `research-${slug || "flow"}`,
        workerIndex: testInfo.workerIndex,
        parallelIndex: testInfo.parallelIndex,
        retry: testInfo.retry,
    });
}

async function seedResearchWorkspace(
    browser: import("@playwright/test").Browser,
    credentials: E2ECredentials
) {
    const isolationToken = `${Date.now()}-${credentials.email.split("@")[0]}`;
    const api = await createAuthenticatedApiContext(browser, credentials, undefined, {
        provision: true,
    });
    try {
        const project = await createProject(api, `E2E Research ${isolationToken}`);
        const corpus = await seedDemoCorpus(api, project.id, `E2E Demo ${isolationToken}`);
        const segmentRun = await segmentCorpus(api, corpus.id, "paragraph");
        if (segmentRun.status !== "completed") {
            await waitForRun(api, segmentRun.id);
        }
        const unitIds = await listTextUnitIdsFromExport(api, corpus.id, "paragraph");
        expect(unitIds.length).toBeGreaterThan(0);

        const codebook = await createCodebook(api, project.id, `E2E Codebook ${isolationToken}`);
        const labels = [];
        for (const name of ["Label A", "Label B", "Label C", "Label D"]) {
            labels.push(await createLabel(api, codebook.id, name));
        }
        expect(labels.length).toBe(4);

        for (let index = 0; index < unitIds.length; index += 1) {
            const unitId = unitIds[index];
            await saveAnnotations(api, {
                text_unit_id: unitId,
                codebook_id: codebook.id,
                values: labels.map((label, labelIndex) => ({
                    label_id: label.id,
                    value: (index + labelIndex) % 2 === 0 ? "yes" : "no",
                })),
            });
        }

        const reliabilityRun = await computeReliability(
            api,
            corpus.id,
            codebook.id,
            labels.map((l) => l.id)
        );
        expect(reliabilityRun.status).toBe("completed");

        const preview = await previewDataset(api, {
            corpus_id: corpus.id,
            unit_type: "paragraph",
            codebook_id: codebook.id,
            label_ids: labels.map((l) => l.id),
        });
        expect(preview.unit_count).toBeGreaterThan(0);

        const snapshot = await freezeDataset(api, {
            name: `E2E training snapshot ${isolationToken}`,
            corpus_id: corpus.id,
            unit_type: "paragraph",
            codebook_id: codebook.id,
            label_ids: labels.map((l) => l.id),
        });

        const trainRun = await trainClassifier(api, snapshot.id);
        expect(trainRun.metrics).toBeTruthy();

        const models = await listClassifiers(api, project.id, corpus.id);
        expect(models.length).toBeGreaterThan(0);

        const predictRun = await predictWithModel(api, models[0].id, "paragraph");
        expect(predictRun.status).toBe("completed");

        const predictions = await listPredictions(api, models[0].id);
        expect(predictions.length).toBeGreaterThan(0);

        const dashboard = await getDashboard(api, corpus.id);
        expect(dashboard.document_count).toBe(4);
        expect(dashboard.trained_model_count).toBeGreaterThan(0);

        return { project, corpus, codebook, models, credentials, api };
    } catch (error) {
        await api.close();
        throw error;
    }
}

async function closeApi(api: ApiContext | undefined): Promise<void> {
    if (api) {
        await api.close();
    }
}

test.describe("Text Research workflow", () => {
    test("full API pipeline: corpus → annotate → train → predict", async ({ browser }, testInfo) => {
        test.setTimeout(120_000);
        const credentials = credentialsForTest(testInfo);
        const seeded = await seedResearchWorkspace(browser, credentials);
        await closeApi(seeded.api);
    });

    test("UI reflects completed research pipeline", async ({ browser }, testInfo) => {
        test.setTimeout(120_000);
        const credentials = credentialsForTest(testInfo);
        const { project, corpus, codebook, models, api } = await seedResearchWorkspace(
            browser,
            credentials
        );

        try {
            // Reuse the already-authenticated session cookies via a fresh UI context
            // signed in as the same isolated user (no shared E2E_TEST_EMAIL).
            const { context } = await createAuthenticatedBrowserContext(browser, credentials, {
                provision: false,
            });
            await context.addInitScript(
                ({ projectId, corpusId, codebookId }) => {
                    localStorage.setItem(`text-research:corpus:${projectId}`, corpusId);
                    localStorage.setItem(`text-research:codebook:${projectId}`, codebookId);
                },
                { projectId: project.id, corpusId: corpus.id, codebookId: codebook.id }
            );

            const page = await context.newPage();

            await page.goto(`/projects/${project.id}`);
            await expect(page.getByRole("button", { name: /Open Text Research/i })).toBeVisible();
            await page.getByRole("button", { name: /Open Text Research/i }).click();
            await expect(page).toHaveURL(new RegExp(`/research/${project.id}/dashboard`));
            await expect(page.getByRole("heading", { name: /Text Research/i })).toBeVisible();
            await expect(page.getByText("Documents").first()).toBeVisible();
            await expect(page.getByText("4").first()).toBeVisible();

            await page.getByRole("tab", { name: "Corpus" }).click();
            await expect(page.getByText(/Showing 4 of 4 documents/i)).toBeVisible({
                timeout: 15_000,
            });

            await page.getByRole("tab", { name: "Classify" }).click();
            await expect(page.getByText(/Trained models/i )).toBeVisible();
            await expect(page.getByText(models[0].name ?? "E2E Classifier")).toBeVisible({
                timeout: 15_000,
            });

            await page.getByRole("tab", { name: "Export" }).click();
            await expect(page.getByText(/export/i).first()).toBeVisible();

            await context.close();
        } finally {
            await closeApi(api);
        }
    });
});
