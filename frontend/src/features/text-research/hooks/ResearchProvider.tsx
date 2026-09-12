import { useCallback, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { listCodebooks, listCorpora, listLabels } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import type { UnitType } from "../types";
import { resolveResearchSelectionId } from "./researchSelection";
import {
    ResearchContext,
    type AskAboutPayload,
    type PendingAsk,
    type ResearchContextValue,
} from "./researchContextState";

const corpusStorageKey = (projectId: string) => `text-research:corpus:${projectId}`;
const codebookStorageKey = (projectId: string) => `text-research:codebook:${projectId}`;

export function ResearchProvider({ children }: { children: React.ReactNode }) {
    const { projectId = "" } = useParams();
    const [corpusSelection, setCorpusSelection] = useState("");
    const [codebookSelection, setCodebookSelection] = useState("");
    const [unitType, setUnitType] = useState<UnitType>("paragraph");
    const [pendingAsk, setPendingAsk] = useState<PendingAsk | null>(null);
    const [askPanelOpen, setAskPanelOpen] = useState(false);

    const askAbout = useCallback((payload: AskAboutPayload) => {
        setPendingAsk({
            question: payload.question,
            intent: payload.intent,
            autoSubmit: payload.autoSubmit ?? true,
            nonce: Date.now(),
        });
        setAskPanelOpen(true);
    }, []);

    const clearPendingAsk = useCallback(() => setPendingAsk(null), []);

    const corporaQuery = useQuery({
        queryKey: queryKeys.textResearch.corpora(projectId),
        queryFn: () => listCorpora(projectId),
        enabled: Boolean(projectId),
        staleTime: QUERY_STALE_TIMES.projects,
    });

    const codebooksQuery = useQuery({
        queryKey: queryKeys.textResearch.codebooks(projectId),
        queryFn: () => listCodebooks(projectId),
        enabled: Boolean(projectId),
        staleTime: QUERY_STALE_TIMES.researchCodebook,
    });

    const corpora = useMemo(() => corporaQuery.data ?? [], [corporaQuery.data]);
    const codebooks = useMemo(() => codebooksQuery.data ?? [], [codebooksQuery.data]);

    const selectedCorpusId = useMemo(() => {
        if (!projectId || corporaQuery.isLoading) return "";
        if (corporaQuery.isFetching && corporaQuery.data === undefined) return corpusSelection;
        return resolveResearchSelectionId({
            projectId,
            storageKey: corpusStorageKey,
            availableIds: corpora.map((corpus) => corpus.id),
            currentId: corpusSelection,
        });
    }, [
        projectId,
        corpora,
        corporaQuery.isLoading,
        corporaQuery.isFetching,
        corporaQuery.data,
        corpusSelection,
    ]);

    const selectedCodebookId = useMemo(() => {
        if (!projectId || codebooksQuery.isLoading) return "";
        if (codebooksQuery.isFetching && codebooksQuery.data === undefined) return codebookSelection;
        return resolveResearchSelectionId({
            projectId,
            storageKey: codebookStorageKey,
            availableIds: codebooks.map((codebook) => codebook.id),
            currentId: codebookSelection,
        });
    }, [
        projectId,
        codebooks,
        codebooksQuery.isLoading,
        codebooksQuery.isFetching,
        codebooksQuery.data,
        codebookSelection,
    ]);

    const setSelectedCorpusId = useCallback(
        (corpusId: string) => {
            setCorpusSelection(corpusId);
            if (!projectId) return;
            if (corpusId) {
                localStorage.setItem(corpusStorageKey(projectId), corpusId);
            } else {
                localStorage.removeItem(corpusStorageKey(projectId));
            }
        },
        [projectId]
    );

    const setSelectedCodebookId = useCallback(
        (codebookId: string) => {
            setCodebookSelection(codebookId);
            if (!projectId) return;
            if (codebookId) {
                localStorage.setItem(codebookStorageKey(projectId), codebookId);
            } else {
                localStorage.removeItem(codebookStorageKey(projectId));
            }
        },
        [projectId]
    );

    const labelsQuery = useQuery({
        queryKey: queryKeys.textResearch.labels(selectedCodebookId),
        queryFn: () => listLabels(selectedCodebookId),
        enabled: Boolean(selectedCodebookId),
        staleTime: QUERY_STALE_TIMES.researchCodebook,
    });

    const value = useMemo<ResearchContextValue>(
        () => ({
            projectId,
            corpora,
            corporaLoading: corporaQuery.isLoading,
            corporaError: corporaQuery.error,
            selectedCorpusId,
            selectedCorpus: corpora.find((c) => c.id === selectedCorpusId) ?? null,
            setSelectedCorpusId,
            codebooks,
            codebooksLoading: codebooksQuery.isLoading,
            selectedCodebookId,
            selectedCodebook: codebooks.find((c) => c.id === selectedCodebookId) ?? null,
            setSelectedCodebookId,
            labels: labelsQuery.data ?? [],
            labelsLoading: labelsQuery.isLoading,
            unitType,
            setUnitType,
            refetchCorpora: () => void corporaQuery.refetch(),
            refetchCodebooks: () => void codebooksQuery.refetch(),
            askAbout,
            pendingAsk,
            clearPendingAsk,
            askPanelOpen,
            setAskPanelOpen,
        }),
        [
            projectId,
            corpora,
            corporaQuery,
            selectedCorpusId,
            setSelectedCorpusId,
            codebooks,
            codebooksQuery,
            selectedCodebookId,
            setSelectedCodebookId,
            labelsQuery.data,
            labelsQuery.isLoading,
            unitType,
            askAbout,
            pendingAsk,
            clearPendingAsk,
            askPanelOpen,
        ]
    );

    return <ResearchContext.Provider value={value}>{children}</ResearchContext.Provider>;
}
