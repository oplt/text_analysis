import { expect, type APIRequestContext, type APIResponse } from "@playwright/test";

const apiBaseUrl = process.env.E2E_API_URL ?? "http://localhost:8000";
const researchBase = `${apiBaseUrl}/api/v1/research`;

export type ApiContext = {
    request: APIRequestContext;
    csrfToken: string | undefined;
    email: string;
    password: string;
    close: () => Promise<void>;
};

export type E2ECredentials = {
    email: string;
    password: string;
};

/** Unique email/password per worker/test so retries never share auth rate-limit keys. */
export function uniqueE2ECredentials(options?: {
    workerIndex?: number;
    parallelIndex?: number;
    retry?: number;
    prefix?: string;
}): E2ECredentials {
    const prefix = options?.prefix ?? "e2e";
    const worker = options?.workerIndex ?? Number(process.env.TEST_WORKER_INDEX ?? "0");
    const parallel = options?.parallelIndex ?? 0;
    const retry = options?.retry ?? 0;
    const stamp = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    const email = `${prefix}-w${worker}-p${parallel}-r${retry}-${stamp}@example.com`;
    const password = process.env.E2E_TEST_PASSWORD ?? `E2E-Pass-${stamp.slice(0, 12)}!Aa1`;
    return { email, password };
}

async function expectOk(response: APIResponse, label: string): Promise<void> {
    if (response.ok()) {
        return;
    }
    const body = await response.text();
    expect(
        response.ok(),
        `${label} failed: HTTP ${response.status()} ${body.slice(0, 500)}`
    ).toBeTruthy();
}

export async function provisionE2EUser(
    request: APIRequestContext,
    credentials: E2ECredentials,
    fullName = "Playwright E2E"
): Promise<void> {
    const signUp = await request.post(`${apiBaseUrl}/api/v1/auth/sign-up`, {
        data: {
            email: credentials.email,
            password: credentials.password,
            full_name: fullName,
        },
    });
    // 202 = created; 409/400 may occur on rare collisions — sign-in will decide.
    if (![202, 400, 409].includes(signUp.status())) {
        await expectOk(signUp, "sign-up");
    }
}

export async function createAuthenticatedApiContext(
    browser: import("@playwright/test").Browser,
    emailOrCredentials: string | E2ECredentials,
    password?: string,
    options?: { provision?: boolean }
): Promise<ApiContext> {
    const credentials: E2ECredentials =
        typeof emailOrCredentials === "string"
            ? { email: emailOrCredentials, password: password ?? "" }
            : emailOrCredentials;
    if (!credentials.email || !credentials.password) {
        throw new Error("E2E credentials require both email and password");
    }

    const context = await browser.newContext();
    const shouldProvision = options?.provision ?? true;
    if (shouldProvision) {
        await provisionE2EUser(context.request, credentials);
    }

    const signIn = await context.request.post(`${apiBaseUrl}/api/v1/auth/sign-in`, {
        data: { email: credentials.email, password: credentials.password },
    });
    await expectOk(signIn, `sign-in (${credentials.email})`);
    const csrfToken = (await context.cookies()).find((cookie) => cookie.name === "csrf_token")
        ?.value;
    return {
        request: context.request,
        csrfToken,
        email: credentials.email,
        password: credentials.password,
        close: () => context.close(),
    };
}

/** Authenticated browser context for UI tests, isolated per credentials. */
export async function createAuthenticatedBrowserContext(
    browser: import("@playwright/test").Browser,
    credentials: E2ECredentials,
    options?: { provision?: boolean }
): Promise<{
    context: import("@playwright/test").BrowserContext;
    email: string;
    password: string;
}> {
    const context = await browser.newContext();
    if (options?.provision ?? false) {
        await provisionE2EUser(context.request, credentials);
    }
    const signIn = await context.request.post(`${apiBaseUrl}/api/v1/auth/sign-in`, {
        data: { email: credentials.email, password: credentials.password },
    });
    await expectOk(signIn, `UI sign-in (${credentials.email})`);
    return { context, email: credentials.email, password: credentials.password };
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
    await expectOk(response, "create project");
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
    await expectOk(response, "seed demo corpus");
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
    await expectOk(response, "segment corpus");
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
        await expectOk(response, `poll run ${runId}`);
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
    await expectOk(response, "export text units");
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
    await expectOk(response, "create codebook");
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
    await expectOk(response, "create label");
    return response.json();
}

export async function listLabels(
    api: ApiContext,
    codebookId: string
): Promise<Array<{ id: string; name: string }>> {
    const response = await api.request.get(`${researchBase}/codebooks/${codebookId}/labels`, {
        headers: headers(api.csrfToken),
    });
    await expectOk(response, "list labels");
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
    await expectOk(response, "save annotations");
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
    await expectOk(response, "compute reliability");
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
    await expectOk(response, "dataset preview");
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
    await expectOk(response, "freeze dataset");
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
    await expectOk(response, "train classifier");
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
    await expectOk(response, "list classifiers");
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
    await expectOk(response, "predict with model");
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
    await expectOk(response, "list predictions");
    return response.json();
}

export async function getDashboard(
    api: ApiContext,
    corpusId: string
): Promise<{ document_count: number; trained_model_count: number }> {
    const response = await api.request.get(`${researchBase}/corpora/${corpusId}/dashboard`, {
        headers: headers(api.csrfToken),
    });
    await expectOk(response, "get dashboard");
    return response.json();
}
