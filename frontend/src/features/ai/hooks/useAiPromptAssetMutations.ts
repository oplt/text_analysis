import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createAiDocument, createPromptTemplate, createPromptVersion, updatePromptTemplate, updatePromptVersion, uploadAiDocument } from "../../../api/ai";
import { useSnackbar } from "../../../app/snackbarContext";
import { invalidateAiOverview, queryKeys } from "../../../config/queryKeys";
import { useMutationErrorToast } from "../../../hooks/useMutationErrorToast";
import type { AiStudioForms } from "./useAiStudioForms";

export function useAiPromptAssetMutations(forms: AiStudioForms) {
    const client = useQueryClient();
    const { showToast } = useSnackbar();
    const toastError = useMutationErrorToast();
    const refreshVersions = () => client.invalidateQueries({
        queryKey: queryKeys.ai.promptVersions(forms.selectedTemplateId),
    });
    const createTemplateMutation = useMutation({ mutationFn: createPromptTemplate,
        onSuccess: async () => { forms.setTemplateForm({ key: "", name: "", description: "" }); await invalidateAiOverview(client); showToast({ message: "Prompt template created.", severity: "success" }); },
        onError: (error) => toastError(error, "Failed to create prompt template.") });
    const createVersionMutation = useMutation({
        mutationFn: ({ templateId, payload }: { templateId: string; payload: Parameters<typeof createPromptVersion>[1] }) => createPromptVersion(templateId, payload),
        onSuccess: async () => { await invalidateAiOverview(client); await refreshVersions(); showToast({ message: "Prompt version created.", severity: "success" }); },
        onError: (error) => toastError(error, "Failed to create prompt version."),
    });
    const activateVersionMutation = useMutation({
        mutationFn: ({ templateId, versionId }: { templateId: string; versionId: string }) => updatePromptTemplate(templateId, { active_version_id: versionId }),
        onSuccess: async () => { await invalidateAiOverview(client); showToast({ message: "Active prompt version updated.", severity: "success" }); },
        onError: (error) => toastError(error, "Failed to update active prompt version."),
    });
    const publishVersionMutation = useMutation({
        mutationFn: ({ templateId, versionId, isPublished }: { templateId: string; versionId: string; isPublished: boolean }) => updatePromptVersion(templateId, versionId, { is_published: isPublished }),
        onSuccess: async () => { await refreshVersions(); showToast({ message: "Prompt version updated.", severity: "success" }); },
        onError: (error) => toastError(error, "Failed to update prompt version."),
    });
    const createTextDocumentMutation = useMutation({ mutationFn: createAiDocument,
        onSuccess: async () => { forms.setTextDocumentForm({ title: "", description: "", content: "", content_type: "text/plain" }); await invalidateAiOverview(client); showToast({ message: "Document queued for indexing.", severity: "success" }); },
        onError: (error) => toastError(error, "Failed to ingest document.") });
    const uploadDocumentMutation = useMutation({
        mutationFn: ({ file, description }: { file: File; description?: string }) => uploadAiDocument(file, description),
        onSuccess: async () => { forms.setUploadDescription(""); await invalidateAiOverview(client); showToast({ message: "Document uploaded and queued for indexing.", severity: "success" }); },
        onError: (error) => toastError(error, "Failed to upload document."),
    });
    return { createTemplateMutation, createVersionMutation, activateVersionMutation,
        publishVersionMutation, createTextDocumentMutation, uploadDocumentMutation };
}
