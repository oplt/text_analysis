import { test, expect } from "@playwright/test";
import {
    computeReliability,
    createAuthenticatedApiContext,
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
    waitForRun,
} from "./helpers/research-api";

const email = process.env.E2E_TEST_EMAIL;
const password = process.env.E2E_TEST_PASSWORD;

async function seedResearchWorkspace(browser: import("@playwright/test").Browser) {
    const api = await createAuthenticatedApiContext(browser, email!, password!);
    try {
        const project = await createProject(api, `E2E Research ${Date.now()}`);
        const corpus = await seedDemoCorpus(api, project.id);
        const segmentRun = await segmentCorpus(api, corpus.id, "paragraph");
        if (segmentRun.status !== "completed") {
            await waitForRun(api, segmentRun.id);
        }
        const unitIds = await listTextUnitIdsFromExport(api, corpus.id, "paragraph");
        expect(unitIds.length).toBeGreaterThan(0);

        const codebook = await createCodebook(api, project.id, `E2E Codebook ${Date.now()}`);
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
            name: "E2E training snapshot",
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

        return { project, corpus, codebook, models };
    } finally {
        await api.close();
    }
}

test.describe("Text Research workflow", () => {
    test.skip(!email || !password, "Set E2E_TEST_EMAIL and E2E_TEST_PASSWORD");

    test("full API pipeline: corpus → annotate → train → predict", async ({ browser }) => {
        test.setTimeout(120_000);
        await seedResearchWorkspace(browser);
    });

    test("UI reflects completed research pipeline", async ({ browser }) => {
        test.setTimeout(120_000);
        const { project, corpus, codebook, models } = await seedResearchWorkspace(browser);

        const context = await browser.newContext();
        await context.addInitScript(
            ({ projectId, corpusId, codebookId }) => {
                localStorage.setItem(`text-research:corpus:${projectId}`, corpusId);
                localStorage.setItem(`text-research:codebook:${projectId}`, codebookId);
            },
            { projectId: project.id, corpusId: corpus.id, codebookId: codebook.id }
        );

        const signIn = await context.request.post(
            `${process.env.E2E_API_URL ?? "http://localhost:8000"}/api/v1/auth/sign-in`,
            { data: { email, password } }
        );
        expect(signIn.ok()).toBeTruthy();

        const page = await context.newPage();

        await page.goto(`/projects/${project.id}`);
        await expect(page.getByRole("button", { name: /Open Text Research/i })).toBeVisible();
        await page.getByRole("button", { name: /Open Text Research/i }).click();
        await expect(page).toHaveURL(new RegExp(`/research/${project.id}/dashboard`));
        await expect(page.getByRole("heading", { name: /Text Research/i })).toBeVisible();
        await expect(page.getByText("Documents").first()).toBeVisible();
        await expect(page.getByText("4").first()).toBeVisible();

        await page.getByRole("tab", { name: "Corpus" }).click();
        await expect(page.getByText(/Showing 4 of 4 documents/i)).toBeVisible({ timeout: 15_000 });

        await page.getByRole("tab", { name: "Classify" }).click();
        await expect(page.getByText(/Trained models/i)).toBeVisible();
        await expect(page.getByText(models[0].name ?? "E2E Classifier")).toBeVisible({
            timeout: 15_000,
        });

        await page.getByRole("tab", { name: "Export" }).click();
        await expect(page.getByText(/export/i).first()).toBeVisible();

        await context.close();
    });
});
