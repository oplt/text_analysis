import { useState } from "react";

export function useAiStudioForms() {
    const [selectedTemplateId, setSelectedTemplateId] = useState("");
    const [selectedDatasetId, setSelectedDatasetId] = useState("");
    const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
    const [selectedEvaluationDocumentIds, setSelectedEvaluationDocumentIds] = useState<string[]>([]);
    const [templateForm, setTemplateForm] = useState({ key: "", name: "", description: "" });
    const [versionForm, setVersionForm] = useState({
        provider_key: "local", model_name: "local-heuristic", system_prompt: "",
        user_prompt_template: "", variable_names: "", response_format: "text" as "text" | "json",
        temperature: "0.2", rollout_percentage: "100", is_published: true,
        input_cost_per_million: "0", output_cost_per_million: "0",
    });
    const [textDocumentForm, setTextDocumentForm] = useState({
        title: "", description: "", content: "", content_type: "text/plain",
    });
    const [uploadDescription, setUploadDescription] = useState("");
    const [runForm, setRunForm] = useState({
        prompt_template_key: "", prompt_version_id: "",
        variables_json: "{\n  \"task\": \"Summarize the attached knowledge base\"\n}",
        retrieval_query: "", top_k: "4", review_required: false,
    });
    const [reviewNotesById, setReviewNotesById] = useState<Record<string, string>>({});
    const [correctionsById, setCorrectionsById] = useState<Record<string, string>>({});
    const [feedbackCommentByRunId, setFeedbackCommentByRunId] = useState<Record<string, string>>({});
    const [datasetForm, setDatasetForm] = useState({ name: "", description: "" });
    const [datasetCaseForm, setDatasetCaseForm] = useState({
        input_variables_json: "{\n  \"task\": \"What is the return policy?\"\n}",
        retrieval_query: "", expected_chunk_ids: "", expected_output_text: "",
        expected_output_json: "", notes: "",
    });

    const toggle = (setter: typeof setSelectedDocumentIds, documentId: string) => setter((current) =>
        current.includes(documentId)
            ? current.filter((item) => item !== documentId)
            : [...current, documentId]
    );
    const toggleDocument = (documentId: string) => toggle(setSelectedDocumentIds, documentId);
    const toggleEvaluationDocument = (documentId: string) =>
        toggle(setSelectedEvaluationDocumentIds, documentId);
    const parseVariableDefinitions = (rawNames: string) => rawNames.split(",")
        .map((item) => item.trim()).filter(Boolean)
        .map((name) => ({ name, description: null, required: true }));

    return { selectedTemplateId, setSelectedTemplateId, selectedDatasetId, setSelectedDatasetId,
        selectedDocumentIds, selectedEvaluationDocumentIds, setSelectedEvaluationDocumentIds,
        templateForm, setTemplateForm, versionForm, setVersionForm, textDocumentForm,
        setTextDocumentForm, uploadDescription, setUploadDescription, runForm, setRunForm,
        reviewNotesById, setReviewNotesById, correctionsById, setCorrectionsById,
        feedbackCommentByRunId, setFeedbackCommentByRunId, datasetForm, setDatasetForm,
        datasetCaseForm, setDatasetCaseForm, parseVariableDefinitions, toggleDocument,
        toggleEvaluationDocument };
}

export type AiStudioForms = ReturnType<typeof useAiStudioForms>;
