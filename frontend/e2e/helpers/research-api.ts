import { expect, type APIRequestContext } from "@playwright/test";

const apiBaseUrl = process.env.E2E_API_URL ?? "http://localhost:8000";
const researchBase = `${apiBaseUrl}/api/v1/research`;

export type ApiContext = {
    request: APIRequestContext;
    csrfToken: string | undefined;
    close: () => Promise<void>;
};

export async function createAuthenticatedApiContext(
    browser: import("@playwright/test").Browser,
    email: string,
    password: string
): Promise<ApiContext> {
    const context = await browser.newContext();
    const signIn = await context.request.post(`${apiBaseUrl}/api/v1/auth/sign-in`, {
        data: { email, password },
    });
    expect(signIn.ok()).toBeTruthy();
    const csrfToken = (await context.cookies()).find((cookie) => cookie.name === "csrf_token")?.value;
    return {
        request: context.request,
        csrfToken,
        close: () => context.close(),
    };
}

function firstCsvField(line: string): string {
    if (line.startsWith('"')) {
        const closing = line.indexOf('"', 1);
        return line.slice(1, closing);
    }
    return line.split(",")[0] ?? "";
}

function headers(csrfToken: string | undefined): Record<string, string> | undefined {
    return csrfToken ? { "X-CSRF-Token": csrfToken } : undefined;
}

export async function createProject(
    api: ApiContext,
    name: string
): Promise<{ id: string; name: string }> {
    const response = await api.request.post(`${apiBaseUrl}/api/v1/projects`, {
        headers: headers(api.csrfToken),
        data: { name, description: "Playwright E2E research workflow" },
    });
    expect(response.ok()).toBeTruthy();
    return response.json();
}

export async function seedDemoCorpus(
    api: ApiContext,
    projectId: string,
    corpusName?: string
): Promise<{ id: string; name: string; project_id: string }> {
    const response = await api.request.post(`${researchBase}/projects/${projectId}/demo-seed`, {
        headers: headers(api.csrfToken),
        data: { corpus_name: corpusName ?? `E2E Demo ${Date.now()}` },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
}

export async function segmentCorpus(
    api: ApiContext,
    corpusId: string,
    unitType: "document" | "paragraph" | "sentence" = "paragraph"
): Promise<{ id: string; status: string }> {
    const response = await api.request.post(`${researchBase}/corpora/${corpusId}/segment`, {
        headers: headers(api.csrfToken),
        data: { unit_type: unitType },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
}

export async function waitForRun(
    api: ApiContext,
    runId: string,
    timeoutMs = 60_000
): Promise<{ id: string; status: string; metrics?: Record<string, unknown> }> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const response = await api.request.get(`${researchBase}/runs/${runId}`, {
            headers: headers(api.csrfToken),
        });
        expect(response.ok()).toBeTruthy();
        const run = await response.json();
        if (run.status === "completed" || run.status === "failed") {
            expect(run.status).toBe("completed");
            return run;
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
    }
    throw new Error(`Run ${runId} did not complete within ${timeoutMs}ms`);
}

export async function listTextUnitIdsFromExport(
    api: ApiContext,
    corpusId: string,
    unitType: "paragraph" | "document" | "sentence" = "paragraph"
): Promise<string[]> {
    const response = await api.request.get(
        `${researchBase}/corpora/${corpusId}/export/units.csv?unit_type=${unitType}`,
        { headers: headers(api.csrfToken) }
    );
    expect(response.ok()).toBeTruthy();
    const csv = await response.text();
    const lines = csv.trim().split("\n");
    expect(lines.length).toBeGreaterThan(1);
    return lines.slice(1).map((line) => firstCsvField(line)).filter(Boolean);
}

export async function createCodebook(
    api: ApiContext,
    projectId: string,
    name: string
): Promise<{ id: string; version: string }> {
    const response = await api.request.post(`${researchBase}/projects/${projectId}/codebooks`, {
        headers: headers(api.csrfToken),
        data: { name, seed_demo_labels: false },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
}

export async function createLabel(
    api: ApiContext,
    codebookId: string,
    name: string,
    description?: string
): Promise<{ id: string; name: string }> {
    const response = await api.request.post(`${researchBase}/codebooks/${codebookId}/labels`, {
        headers: headers(api.csrfToken),
        data: { name, description: description ?? `Synthetic fixture label ${name}` },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
}

export async function listLabels(
    api: ApiContext,
    codebookId: string
): Promise<Array<{ id: string; name: string }>> {
    const response = await api.request.get(`${researchBase}/codebooks/${codebookId}/labels`, {
        headers: headers(api.csrfToken),
    });
    expect(response.ok()).toBeTruthy();
    return response.json();
}

export async function saveAnnotations(
    api: ApiContext,
    payload: {
        text_unit_id: string;
        codebook_id: string;
        values: Array<{ label_id: string; value: string }>;
    }
): Promise<void> {
    const response = await api.request.post(`${researchBase}/annotations`, {
        headers: headers(api.csrfToken),
        data: { ...payload, mark_task_complete: true },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
}

export async function computeReliability(
    api: ApiContext,
    corpusId: string,
    codebookId: string,
    labelIds: string[]
): Promise<{ id: string; status: string }> {
    const response = await api.request.post(`${researchBase}/corpora/${corpusId}/reliability`, {
        headers: headers(api.csrfToken),
        data: { codebook_id: codebookId, label_ids: labelIds },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
}

export async function previewDataset(
    api: ApiContext,
    payload: {
        corpus_id: string;
        unit_type: string;
        codebook_id: string;
        label_ids: string[];
    }
): Promise<{ unit_count: number; warnings: string[] }> {
    const response = await api.request.post(`${researchBase}/classifiers/dataset-preview`, {
        headers: headers(api.csrfToken),
        data: { annotation_source: "adjudicated_only", ...payload },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
}

export async function freezeDataset(
    api: ApiContext,
    payload: {
        name: string;
        corpus_id: string;
        unit_type: string;
        codebook_id: string;
        label_ids: string[];
    }
): Promise<{ id: string }> {
    const response = await api.request.post(`${researchBase}/classifiers/dataset-snapshots`, {
        headers: headers(api.csrfToken),
        data: { annotation_source: "adjudicated_only", ...payload },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
}

export async function trainClassifier(
    api: ApiContext,
    snapshotId: string
): Promise<{ id: string; status: string; metrics?: Record<string, unknown> }> {
    const response = await api.request.post(`${researchBase}/classifiers/train`, {
        headers: headers(api.csrfToken),
        data: {
            snapshot_id: snapshotId,
            algorithm: "logistic_regression",
            run_async: false,
            name: "E2E Classifier",
        },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    const run = await response.json();
    if (run.status !== "completed") {
        return waitForRun(api, run.id);
    }
    return run;
}

export async function listClassifiers(
    api: ApiContext,
    projectId: string,
    corpusId: string
): Promise<Array<{ id: string; name: string | null; metrics: Record<string, unknown> }>> {
    const response = await api.request.get(
        `${researchBase}/projects/${projectId}/classifiers?corpus_id=${encodeURIComponent(corpusId)}`,
        { headers: headers(api.csrfToken) }
    );
    expect(response.ok()).toBeTruthy();
    return response.json();
}

export async function predictWithModel(
    api: ApiContext,
    modelId: string,
    unitType: "paragraph" | "document" | "sentence" = "paragraph"
): Promise<{ id: string; status: string }> {
    const response = await api.request.post(`${researchBase}/classifiers/${modelId}/predict`, {
        headers: headers(api.csrfToken),
        data: { unit_type: unitType, only_unannotated: false },
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    const run = await response.json();
    if (run.status !== "completed") {
        return waitForRun(api, run.id);
    }
    return run;
}

export async function listPredictions(
    api: ApiContext,
    modelId: string
): Promise<unknown[]> {
    const response = await api.request.get(`${researchBase}/classifiers/${modelId}/predictions`, {
        headers: headers(api.csrfToken),
    });
    expect(response.ok()).toBeTruthy();
    return response.json();
}

export async function getDashboard(
    api: ApiContext,
    corpusId: string
): Promise<{ document_count: number; trained_model_count: number }> {
    const response = await api.request.get(`${researchBase}/corpora/${corpusId}/dashboard`, {
        headers: headers(api.csrfToken),
    });
    expect(response.ok()).toBeTruthy();
    return response.json();
}
