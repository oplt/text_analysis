import { describe, expect, it } from "vitest";
import { buildProvenanceSections, provenanceRawPayload } from "./provenanceModel";

describe("provenanceModel", () => {
    it("builds friendly sections from run parameters", () => {
        const sections = buildProvenanceSections({
            run: {
                id: "run-1",
                run_type: "frequency",
                status: "completed",
                corpus_id: "corpus-1",
                parameters: {
                    unit_type: "paragraph",
                    preprocessing_profile_id: "prep-1",
                    codebook_id: "cb-1",
                    codebook_version: "2",
                    dataset_snapshot_id: "snap-1",
                    algorithm: "logistic_regression",
                    random_seed: 42,
                },
                random_seed: 42,
                artifact_path: "/artifacts/run-1",
                created_by: "user-1",
                created_at: "2026-01-01T00:00:00.000Z",
                started_at: "2026-01-01T00:01:00.000Z",
                completed_at: "2026-01-01T00:02:00.000Z",
                evidence_revision_hash: "abc123",
                project_id: "proj-1",
            },
        });

        const titles = sections.map((section) => section.title);
        expect(titles).toContain("Run identity");
        expect(titles).toContain("Data");
        expect(titles).toContain("Preprocessing");
        expect(titles).toContain("Measurement");
        expect(titles).toContain("Model & seed");
        expect(titles).toContain("Source revisions & checksums");

        const identity = sections.find((section) => section.id === "identity");
        expect(identity?.fields.some((field) => field.label === "Run ID")).toBe(true);
        expect(identity?.fields.some((field) => field.label === "Created by")).toBe(true);

        const model = sections.find((section) => section.id === "model");
        expect(model?.fields.some((field) => field.label === "Random seed" && field.value === "42")).toBe(
            true
        );
        expect(
            model?.fields.some((field) => field.label === "Dataset snapshot" && field.value === "snap-1")
        ).toBe(true);
    });

    it("merges /provenance detail checksums and reproduce fields", () => {
        const sections = buildProvenanceSections({
            run: {
                id: "run-2",
                run_type: "reliability",
                status: "completed",
                corpus_id: "c1",
                parameters: {},
                random_seed: null,
                artifact_path: null,
                created_by: "u1",
                created_at: "2026-01-01T00:00:00.000Z",
                started_at: null,
                completed_at: null,
                evidence_revision_hash: null,
            },
            detail: {
                run_id: "run-2",
                run_type: "reliability",
                status: "completed",
                random_seed: 7,
                artifact_path: null,
                created_by: "u1",
                corpus_id: "c1",
                project_id: "p1",
                provenance: {
                    corpus_checksum: "corp-hash",
                    pipeline_checksum: "pipe-hash",
                    git_commit: "deadbeef",
                },
                reproduce: {
                    analysis_spec_hash: "spec-hash",
                    exact_reproducible: true,
                    replayable: true,
                },
                runtime_now: {},
            },
        });

        const revisions = sections.find((section) => section.id === "revisions");
        expect(revisions?.fields.some((field) => field.label === "Corpus checksum")).toBe(true);

        const reproduce = sections.find((section) => section.id === "reproduce");
        expect(reproduce?.fields.some((field) => field.label === "Analysis spec hash")).toBe(true);

        const raw = provenanceRawPayload({
            run: {
                id: "run-2",
                run_type: "reliability",
                status: "completed",
                corpus_id: "c1",
                parameters: {},
                random_seed: null,
                artifact_path: null,
                created_by: "u1",
                created_at: "2026-01-01T00:00:00.000Z",
                started_at: null,
                completed_at: null,
                evidence_revision_hash: null,
            },
            detail: null,
        });
        expect(raw.run).toMatchObject({ id: "run-2" });
    });
});
