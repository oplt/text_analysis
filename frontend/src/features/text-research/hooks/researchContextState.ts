import { createContext, useContext } from "react";
import type { AnnotationLabel, Codebook, ResearchCorpus, UnitType } from "../types";

export type AskAboutPayload = {
    question: string;
    intent?: string;
    autoSubmit?: boolean;
};

export type PendingAsk = AskAboutPayload & { nonce: number };

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
    /** Open Ask Corpus panel and optionally prefill/auto-submit a question. */
    askAbout: (payload: AskAboutPayload) => void;
    pendingAsk: PendingAsk | null;
    clearPendingAsk: () => void;
    askPanelOpenNonce: number;
};

export const ResearchContext = createContext<ResearchContextValue | null>(null);

export function useResearchContext() {
    const ctx = useContext(ResearchContext);
    if (!ctx) {
        throw new Error("useResearchContext must be used within ResearchProvider");
    }
    return ctx;
}
