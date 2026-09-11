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
    getRun,
    listAnalysisEngines,
    listClassifiers,
    listPredictions,
    listTextUnitIdsFromExport,
    predictWithModel,
    previewDataset,
    runFrequencies,
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
 *
 * Retries (playwright.config) cover browser flakiness only — waitForRun fails
 * immediately on terminal application failures with run diagnostics.
 */
test.describe.configure({ mode: "serial" });

/** Opt-in R path: requires RESEARCH_R_ENABLED + a live research_r worker. */
const rE2EEnabled = ["1", "true", "yes"].includes(
    (process.env.E2E_R_ENABLED ?? "").trim().toLowerCase()
);

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

            const workflow = page.getByRole("navigation", { name: "Research workflow" });

            const corpusStage = workflow.getByRole("button", { name: /^Corpus\b/ });
            await expect(corpusStage).toBeVisible({ timeout: 15_000 });
            await corpusStage.click();
            await expect(page).toHaveURL(new RegExp(`/research/${project.id}/corpus`));
            await expect(page.getByText(/Showing 4 of 4 documents/i)).toBeVisible({
                timeout: 15_000,
            });

            const classifyStage = workflow.getByRole("button", { name: /^Classify\b/ });
            await expect(classifyStage).toBeVisible({ timeout: 15_000 });
            await classifyStage.click();
            await expect(page).toHaveURL(new RegExp(`/research/${project.id}/classification`));
            await expect(page.getByText(/Trained models/i)).toBeVisible();
            await expect(page.getByText(models[0].name ?? "E2E Classifier")).toBeVisible({
                timeout: 15_000,
            });

            const exportStage = workflow.getByRole("button", { name: /^Export\b/ });
            await expect(exportStage).toBeVisible({ timeout: 15_000 });
            await exportStage.click();
            await expect(page).toHaveURL(new RegExp(`/research/${project.id}/exports`));
            await expect(page.getByText(/export/i).first()).toBeVisible();

            await context.close();
        } finally {
            await closeApi(api);
        }
    });

    test("API frequencies complete with diagnostic polling", async ({ browser }, testInfo) => {
        // Keep under the file's existing budget — do not inflate timeouts to hide failures.
        test.setTimeout(90_000);
        const credentials = credentialsForTest(testInfo);
        const api = await createAuthenticatedApiContext(browser, credentials, undefined, {
            provision: true,
        });
        try {
            const project = await createProject(api, `E2E Freq ${Date.now()}`);
            const corpus = await seedDemoCorpus(api, project.id);
            const segmentRun = await segmentCorpus(api, corpus.id, "paragraph");
            if (segmentRun.status !== "completed") {
                await waitForRun(api, segmentRun.id);
            }
            const run = await runFrequencies(api, corpus.id, {
                unit_type: "paragraph",
                engine: { runtime: "python" },
                run_async: true,
            });
            expect(run.status).toBe("completed");
            expect(run.results).toBeTruthy();
            const persisted = await getRun(api, run.id);
            expect(persisted.status).toBe("completed");
            expect(persisted.error_message ?? null).toBeNull();
        } finally {
            await closeApi(api);
        }
    });

    test("R frequencies when worker ready", async ({ browser }, testInfo) => {
        test.skip(
            !rE2EEnabled,
            "Set E2E_R_ENABLED=1 with RESEARCH_R_ENABLED and a live research_r worker"
        );
        test.setTimeout(120_000);
        const credentials = credentialsForTest(testInfo);
        const api = await createAuthenticatedApiContext(browser, credentials, undefined, {
            provision: true,
        });
        try {
            const engines = await listAnalysisEngines(api);
            const rEngine = engines.engines.find((engine) => engine.name === "r");
            expect(rEngine, "R engine missing from /analysis-engines").toBeTruthy();
            expect(rEngine?.available).toBe(true);
            expect(rEngine?.ready).toBe(true);
            expect(rEngine?.analyses ?? []).toContain("frequencies");

            const project = await createProject(api, `E2E R ${Date.now()}`);
            const corpus = await seedDemoCorpus(api, project.id);
            const segmentRun = await segmentCorpus(api, corpus.id, "paragraph");
            if (segmentRun.status !== "completed") {
                await waitForRun(api, segmentRun.id);
            }

            const run = await runFrequencies(api, corpus.id, {
                unit_type: "paragraph",
                engine: { runtime: "r", preprocessing_mode: "standardized" },
                run_async: true,
            });
            expect(run.status).toBe("completed");
            expect(run.results).toBeTruthy();

            const persisted = await getRun(api, run.id);
            expect(persisted.status).toBe("completed");
            const provenance = (
                persisted.parameters as { provenance?: Record<string, unknown> } | undefined
            )?.provenance;
            const analysisSpec = provenance?.analysis_specification as
                | { engine?: { runtime?: string } }
                | undefined;
            expect(
                analysisSpec?.engine?.runtime,
                `expected R engine in run ${persisted.id}; provenance keys=${Object.keys(provenance ?? {}).join(",")}`
            ).toBe("r");
            expect(String(provenance?.engine_version ?? "")).toMatch(/r-quanteda/);
            expect(rEngine?.implementation_version).toBeTruthy();
            expect(Object.keys(persisted.results ?? {}).length).toBeGreaterThan(0);        } finally {
            await closeApi(api);
        }
    });
});
