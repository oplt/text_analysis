/**
 * Shared scientific provenance model (Phase 19).
 * Friendly labels for run + /provenance payloads; raw retained separately.
 */

import type { HelpTermId } from "../../config/helpText";
import { formatDisplayValue } from "../../components/ui/jsonDisplay";
import type { RunProvenance } from "../../api/textResearch";

export type ProvenanceField = {
    key: string;
    label: string;
    value: string;
    helpTermId?: HelpTermId;
};

export type ProvenanceSection = {
    id: string;
    title: string;
    fields: ProvenanceField[];
};

const PARAM_LABELS: Record<string, string> = {
    corpus_id: "Corpus",
    document_ids: "Documents",
    unit_type: "Unit type",
    filters: "Filters",
    preprocessing_profile_id: "Preprocessing profile",
    preprocessing_config: "Preprocessing config",
    codebook_id: "Codebook",
    codebook_version: "Codebook version",
    dataset_snapshot_id: "Dataset snapshot",
    annotation_source: "Annotation source",
    minimum_agreement: "Minimum agreement",
    provenance_mode: "Provenance mode",
    algorithm: "Algorithm / model family",
    model_id: "Model ID",
    trained_model_id: "Trained model",
    random_seed: "Random seed",
    test_size: "Test size",
    class_weight: "Class weight",
    C: "Regularization (C)",
    grouped_split: "Grouped split",
    ngram_range: "N-gram range",
    min_df: "Min document frequency",
    max_df: "Max document frequency",
};

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
}

function field(
    key: string,
    label: string,
    value: unknown,
    helpTermId?: HelpTermId
): ProvenanceField | null {
    if (value == null || value === "") return null;
    const display = formatDisplayValue(value);
    if (display === "—" || display === "{}" || display === "[]") return null;
    return { key, label, value: display, helpTermId };
}

function pushField(fields: ProvenanceField[], next: ProvenanceField | null) {
    if (next) fields.push(next);
}

function paramFields(
    parameters: Record<string, unknown> | null | undefined,
    keys: string[]
): ProvenanceField[] {
    const fields: ProvenanceField[] = [];
    for (const key of keys) {
        const value = parameters?.[key];
        pushField(
            fields,
            field(key, PARAM_LABELS[key] ?? key.replaceAll("_", " "), value)
        );
    }
    return fields;
}

function checksumFields(provenance: Record<string, unknown> | null): ProvenanceField[] {
    if (!provenance) return [];
    const fields: ProvenanceField[] = [];
    pushField(
        fields,
        field("corpus_checksum", "Corpus checksum", provenance.corpus_checksum, "provenance")
    );
    pushField(
        fields,
        field(
            "pipeline_checksum",
            "Pipeline checksum",
            provenance.pipeline_checksum,
            "provenance"
        )
    );
    pushField(
        fields,
        field(
            "evidence_revision_hash",
            "Evidence revision",
            provenance.evidence_revision_hash ?? provenance.source_revision,
            "provenance"
        )
    );
    pushField(fields, field("git_commit", "Git commit", provenance.git_commit));
    pushField(fields, field("nlp_model", "NLP model", provenance.nlp_model));
    pushField(
        fields,
        field("package_versions", "Package versions", provenance.package_versions)
    );
    pushField(fields, field("random_seeds", "Recorded seeds", provenance.random_seeds));
    pushField(
        fields,
        field(
            "container_image_digest",
            "Container image digest",
            provenance.container_image_digest
        )
    );
    pushField(
        fields,
        field(
            "implementation_version",
            "Implementation version",
            provenance.implementation_version
        )
    );
    pushField(
        fields,
        field(
            "parent_artifact_checksums",
            "Parent artifact checksums",
            provenance.parent_artifact_checksums
        )
    );
    pushField(
        fields,
        field(
            "synthesis_provenance",
            "Synthesis provenance",
            provenance.synthesis_provenance
        )
    );
    return fields;
}

function reproduceFields(reproduce: Record<string, unknown> | null): ProvenanceField[] {
    if (!reproduce) return [];
    const fields: ProvenanceField[] = [];
    pushField(
        fields,
        field(
            "analysis_spec_hash",
            "Analysis spec hash",
            reproduce.analysis_spec_hash,
            "provenance"
        )
    );
    pushField(
        fields,
        field("exact_reproducible", "Exact reproducible", reproduce.exact_reproducible)
    );
    pushField(fields, field("replayable", "Replayable", reproduce.replayable));
    pushField(
        fields,
        field("block_reason", "Reproduce block reason", reproduce.block_reason)
    );
    pushField(
        fields,
        field(
            "analysis_specification",
            "Analysis specification",
            reproduce.analysis_specification
        )
    );
    // Remaining keys as friendly leftovers
    for (const [key, value] of Object.entries(reproduce)) {
        if (
            [
                "analysis_spec_hash",
                "exact_reproducible",
                "replayable",
                "block_reason",
                "analysis_specification",
                "parameters",
            ].includes(key)
        ) {
            continue;
        }
        pushField(fields, field(key, key.replaceAll("_", " "), value));
    }
    return fields;
}

export type ProvenanceRunLike = {
    id: string;
    run_type: string;
    status: string;
    corpus_id?: string | null;
    project_id?: string | null;
    parameters?: Record<string, unknown> | null;
    random_seed?: number | null;
    artifact_path?: string | null;
    created_by?: string | null;
    created_at?: string | null;
    started_at?: string | null;
    completed_at?: string | null;
    evidence_revision_hash?: string | null;
};

/**
 * Build labeled provenance sections from an AnalysisRun and optional /provenance payload.
 */
export function buildProvenanceSections(options: {
    run?: ProvenanceRunLike | null;
    detail?: RunProvenance | null;
}): ProvenanceSection[] {
    const run = options.run ?? null;
    const detail = options.detail ?? null;
    const parameters =
        asRecord(run?.parameters) ??
        asRecord(detail?.reproduce?.parameters) ??
        asRecord(detail?.provenance?.parameters) ??
        {};
    const sections: ProvenanceSection[] = [];

    const identity: ProvenanceField[] = [];
    pushField(identity, field("run_id", "Run ID", run?.id ?? detail?.run_id, "provenance"));
    pushField(identity, field("run_type", "Run type", run?.run_type ?? detail?.run_type));
    pushField(identity, field("status", "Status", run?.status ?? detail?.status));
    pushField(
        identity,
        field("project_id", "Project", run?.project_id ?? detail?.project_id)
    );
    pushField(
        identity,
        field("created_by", "Created by", run?.created_by ?? detail?.created_by)
    );
    pushField(
        identity,
        field(
            "created_at",
            "Created",
            run?.created_at
                ? new Date(run.created_at).toLocaleString()
                : null
        )
    );
    pushField(
        identity,
        field(
            "started_at",
            "Started",
            run?.started_at
                ? new Date(run.started_at).toLocaleString()
                : detail?.started_at
                  ? new Date(detail.started_at).toLocaleString()
                  : null
        )
    );
    pushField(
        identity,
        field(
            "completed_at",
            "Completed",
            run?.completed_at
                ? new Date(run.completed_at).toLocaleString()
                : detail?.completed_at
                  ? new Date(detail.completed_at).toLocaleString()
                  : null
        )
    );
    if (identity.length) {
        sections.push({ id: "identity", title: "Run identity", fields: identity });
    }

    const data = [
        ...paramFields(parameters, ["corpus_id", "document_ids", "unit_type", "filters"]),
    ];
    pushField(
        data,
        field("corpus_id_run", "Corpus", run?.corpus_id ?? detail?.corpus_id)
    );
    // Deduplicate corpus if both present with same value
    const seen = new Set<string>();
    const dataDedup = data.filter((item) => {
        const sig = `${item.label}:${item.value}`;
        if (seen.has(sig)) return false;
        seen.add(sig);
        return true;
    });
    if (dataDedup.length) {
        sections.push({ id: "data", title: "Data", fields: dataDedup });
    }

    const preprocessing = paramFields(parameters, [
        "preprocessing_profile_id",
        "preprocessing_config",
        "ngram_range",
        "min_df",
        "max_df",
    ]);
    if (preprocessing.length) {
        sections.push({
            id: "preprocessing",
            title: "Preprocessing",
            fields: preprocessing,
        });
    }

    const measurement = paramFields(parameters, [
        "codebook_id",
        "codebook_version",
        "annotation_source",
        "minimum_agreement",
        "provenance_mode",
    ]);
    if (measurement.length) {
        sections.push({ id: "measurement", title: "Measurement", fields: measurement });
    }

    const model = paramFields(parameters, [
        "algorithm",
        "dataset_snapshot_id",
        "model_id",
        "trained_model_id",
        "test_size",
        "class_weight",
        "C",
        "grouped_split",
    ]);
    pushField(
        model,
        field(
            "random_seed",
            "Random seed",
            run?.random_seed ?? detail?.random_seed ?? parameters.random_seed,
            "provenance"
        )
    );
    pushField(model, field("artifact_path", "Artifact", run?.artifact_path ?? detail?.artifact_path));
    if (model.length) {
        sections.push({ id: "model", title: "Model & seed", fields: model });
    }

    const revisions: ProvenanceField[] = [];
    pushField(
        revisions,
        field(
            "evidence_revision_hash",
            "Evidence revision",
            run?.evidence_revision_hash,
            "provenance"
        )
    );
    revisions.push(...checksumFields(asRecord(detail?.provenance)));
    if (revisions.length) {
        sections.push({
            id: "revisions",
            title: "Source revisions & checksums",
            fields: revisions,
        });
    }

    const reproduce = reproduceFields(asRecord(detail?.reproduce));
    if (reproduce.length) {
        sections.push({
            id: "reproduce",
            title: "Reproducibility",
            fields: reproduce,
            });
    }

    return sections;
}

export function provenanceRawPayload(options: {
    run?: ProvenanceRunLike | null;
    detail?: RunProvenance | null;
}): Record<string, unknown> {
    return {
        run: options.run
            ? {
                  id: options.run.id,
                  run_type: options.run.run_type,
                  status: options.run.status,
                  corpus_id: options.run.corpus_id,
                  parameters: options.run.parameters,
                  random_seed: options.run.random_seed,
                  artifact_path: options.run.artifact_path,
                  created_by: options.run.created_by,
                  created_at: options.run.created_at,
                  evidence_revision_hash: options.run.evidence_revision_hash,
              }
            : null,
        detail: options.detail ?? null,
    };
}
