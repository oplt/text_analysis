import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { listCodebooks, listCorpora, listLabels } from "../../../api/textResearch";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import type { AnnotationLabel, Codebook, ResearchCorpus, UnitType } from "../types";
import { resolveResearchSelectionId } from "./researchSelection";

const corpusStorageKey = (projectId: string) => `text-research:corpus:${projectId}`;
const codebookStorageKey = (projectId: string) => `text-research:codebook:${projectId}`;

type ResearchContextValue = {
    projectId: string;
    corpora: ResearchCorpus[];
    corporaLoading: boolean;
    corporaError: unknown;
    selectedCorpusId: string;
    selectedCorpus: ResearchCorpus | null;
    setSelectedCorpusId: (corpusId: string) => void;
    codebooks: Codebook[];
    codebooksLoading: boolean;
    selectedCodebookId: string;
    selectedCodebook: Codebook | null;
    setSelectedCodebookId: (codebookId: string) => void;
    labels: AnnotationLabel[];
    labelsLoading: boolean;
    unitType: UnitType;
    setUnitType: (unitType: UnitType) => void;
    refetchCorpora: () => void;
    refetchCodebooks: () => void;
};

const ResearchContext = createContext<ResearchContextValue | null>(null);

export function ResearchProvider({ children }: { children: React.ReactNode }) {
    const { projectId = "" } = useParams();
    const [selectedCorpusId, setSelectedCorpusIdState] = useState("");
    const [selectedCodebookId, setSelectedCodebookIdState] = useState("");
    const [unitType, setUnitType] = useState<UnitType>("paragraph");

    const corporaQuery = useQuery({
        queryKey: queryKeys.textResearch.corpora(projectId),
        queryFn: () => listCorpora(projectId),
        enabled: Boolean(projectId),
        staleTime: QUERY_STALE_TIMES.researchReference,
    });

    const codebooksQuery = useQuery({
        queryKey: queryKeys.textResearch.codebooks(projectId),
        queryFn: () => listCodebooks(projectId),
        enabled: Boolean(projectId),
        staleTime: QUERY_STALE_TIMES.researchReference,
    });

    const corpora = corporaQuery.data ?? [];
    const codebooks = codebooksQuery.data ?? [];

    useEffect(() => {
        if (!projectId || corporaQuery.isLoading) return;
        if (corporaQuery.isFetching && corporaQuery.data === undefined) return;

        setSelectedCorpusIdState((current) => {
            const next = resolveResearchSelectionId({
                projectId,
                storageKey: corpusStorageKey,
                availableIds: corpora.map((corpus) => corpus.id),
                currentId: current,
            });
            return current === next ? current : next;
        });
    }, [projectId, corpora, corporaQuery.isLoading, corporaQuery.isFetching, corporaQuery.data]);

    useEffect(() => {
        if (!projectId || codebooksQuery.isLoading) return;
        if (codebooksQuery.isFetching && codebooksQuery.data === undefined) return;

        setSelectedCodebookIdState((current) => {
            const next = resolveResearchSelectionId({
                projectId,
                storageKey: codebookStorageKey,
                availableIds: codebooks.map((codebook) => codebook.id),
                currentId: current,
            });
            return current === next ? current : next;
        });
    }, [
        projectId,
        codebooks,
        codebooksQuery.isLoading,
        codebooksQuery.isFetching,
        codebooksQuery.data,
    ]);

    const setSelectedCorpusId = useCallback(
        (corpusId: string) => {
            setSelectedCorpusIdState(corpusId);
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
            setSelectedCodebookIdState(codebookId);
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
        staleTime: QUERY_STALE_TIMES.researchReference,
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
        }),
        [
            projectId,
            corpora,
            corporaQuery.isLoading,
            corporaQuery.error,
            selectedCorpusId,
            setSelectedCorpusId,
            codebooks,
            codebooksQuery.isLoading,
            selectedCodebookId,
            setSelectedCodebookId,
            labelsQuery.data,
            labelsQuery.isLoading,
            unitType,
            corporaQuery,
            codebooksQuery,
        ]
    );

    return <ResearchContext.Provider value={value}>{children}</ResearchContext.Provider>;
}

export function useResearchContext() {
    const ctx = useContext(ResearchContext);
    if (!ctx) {
        throw new Error("useResearchContext must be used within ResearchProvider");
    }
    return ctx;
}
