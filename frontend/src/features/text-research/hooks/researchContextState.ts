import { createContext, useContext } from "react";
import type { AnnotationLabel, Codebook, ResearchCorpus, UnitType } from "../types";

export type ResearchContextValue = {
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

export const ResearchContext = createContext<ResearchContextValue | null>(null);

export function useResearchContext() {
    const ctx = useContext(ResearchContext);
    if (!ctx) {
        throw new Error("useResearchContext must be used within ResearchProvider");
    }
    return ctx;
}
