import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { ResearchContext } from "../hooks/researchContextState";
import type { ResearchContextValue } from "../hooks/researchContextState";
import { ResearchContextBar } from "./ResearchContextBar";

const showToast = vi.fn();

vi.mock("../../../app/snackbarContext", () => ({
    useSnackbar: () => ({ showToast }),
}));

function makeCtx(partial: Partial<ResearchContextValue> = {}): ResearchContextValue {
    return {
        projectId: "p1",
        corpora: [
            {
                id: "c1",
                project_id: "p1",
                name: "Corpus A",
                description: null,
                created_by: "u1",
                created_at: "",
                updated_at: "",
            },
            {
                id: "c2",
                project_id: "p1",
                name: "Corpus B",
                description: null,
                created_by: "u1",
                created_at: "",
                updated_at: "",
            },
        ],
        corporaLoading: false,
        corporaError: null,
        selectedCorpusId: "c1",
        selectedCorpus: {
            id: "c1",
            project_id: "p1",
            name: "Corpus A",
            description: null,
            created_by: "u1",
            created_at: "",
            updated_at: "",
        },
        setSelectedCorpusId: vi.fn(),
        codebooks: [
            {
                id: "cb1",
                project_id: "p1",
                name: "Main",
                description: null,
                version: "1",
                is_frozen: false,
                created_by: "u1",
                created_at: "",
            },
            {
                id: "cb2",
                project_id: "p1",
                name: "Alt",
                description: null,
                version: "2",
                is_frozen: true,
                created_by: "u1",
                created_at: "",
            },
        ],
        codebooksLoading: false,
        selectedCodebookId: "cb1",
        selectedCodebook: {
            id: "cb1",
            project_id: "p1",
            name: "Main",
            description: null,
            version: "1",
            is_frozen: false,
            created_by: "u1",
            created_at: "",
        },
        setSelectedCodebookId: vi.fn(),
        labels: [],
        labelsLoading: false,
        unitType: "paragraph",
        setUnitType: vi.fn(),
        refetchCorpora: vi.fn(),
        refetchCodebooks: vi.fn(),
        askAbout: vi.fn(),
        pendingAsk: null,
        clearPendingAsk: vi.fn(),
        askPanelOpen: false,
        setAskPanelOpen: vi.fn(),
        ...partial,
    };
}

function renderBar(ctx: ResearchContextValue, props: { variant?: "full" | "summary"; projectName?: string } = {}) {
    return render(
        <ThemeProvider theme={createTheme()}>
            <MemoryRouter>
                <ResearchContext.Provider value={ctx}>
                    <ResearchContextBar projectName={props.projectName ?? "Demo Project"} variant={props.variant} />
                </ResearchContext.Provider>
            </MemoryRouter>
        </ThemeProvider>
    );
}

describe("ResearchContextBar", () => {
    beforeEach(() => {
        showToast.mockReset();
    });

    it("renders project, corpus, unit, and codebook controls", () => {
        renderBar(makeCtx());
        expect(screen.getByRole("region", { name: "Research context" })).toBeInTheDocument();
        expect(screen.getByRole("link", { name: /Demo Project/i })).toBeInTheDocument();
        expect(screen.getByLabelText("Corpus")).toBeInTheDocument();
        expect(screen.getByLabelText("Unit type")).toBeInTheDocument();
        expect(screen.getByLabelText("Codebook")).toBeInTheDocument();
    });

    it("confirms before switching corpus when one is already selected", async () => {
        const user = userEvent.setup();
        const setSelectedCorpusId = vi.fn();
        renderBar(makeCtx({ setSelectedCorpusId }));

        await user.click(screen.getByLabelText("Corpus"));
        await user.click(await screen.findByRole("option", { name: "Corpus B" }));

        expect(setSelectedCorpusId).not.toHaveBeenCalled();
        const dialog = screen.getByRole("dialog");
        expect(within(dialog).getByText(/Switch corpus to “Corpus B”/i)).toBeInTheDocument();

        await user.click(within(dialog).getByRole("button", { name: "Change" }));
        expect(setSelectedCorpusId).toHaveBeenCalledWith("c2");
        expect(showToast).toHaveBeenCalled();
    });

    it("renders read-only summary chips without editors", () => {
        renderBar(makeCtx(), { variant: "summary" });
        expect(screen.getByLabelText("Research context summary")).toBeInTheDocument();
        expect(screen.getByText(/Corpus: Corpus A/)).toBeInTheDocument();
        expect(screen.queryByLabelText("Corpus")).not.toBeInTheDocument();
    });
});
