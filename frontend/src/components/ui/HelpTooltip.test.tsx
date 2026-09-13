import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { HelpFieldLabel, HelpTooltip } from "./HelpTooltip";
import { HELP_TERMS, getHelpTerm, helpTermIdForKey } from "../../config/helpText";

function wrap(ui: React.ReactElement) {
    return render(<ThemeProvider theme={createTheme()}>{ui}</ThemeProvider>);
}

describe("helpText registry", () => {
    it("covers every Phase 5 required term", () => {
        const required = [
            "cohens_kappa",
            "fleiss_kappa",
            "krippendorff_alpha",
            "precision",
            "recall",
            "f1",
            "macro_f1",
            "micro_f1",
            "tfidf",
            "vocabulary_size",
            "min_df",
            "max_df",
            "dimensionality",
            "confidence",
            "annotation_agreement",
            "chunk_size",
            "chunk_overlap",
            "embeddings",
            "similarity_threshold",
            "top_k_retrieval",
            "reranking",
            "temperature",
            "model_lifecycle",
            "active_learning",
            "drift",
            "robustness",
            "provenance",
        ] as const;
        for (const id of required) {
            expect(getHelpTerm(id)?.definition.length).toBeGreaterThan(10);
            expect(HELP_TERMS[id].id).toBe(id);
        }
    });

    it("maps metric aliases to term ids", () => {
        expect(helpTermIdForKey("mean_cohens_kappa")).toBe("cohens_kappa");
        expect(helpTermIdForKey("f1_macro")).toBe("macro_f1");
        expect(helpTermIdForKey("min_df")).toBe("min_df");
    });
});

describe("HelpTooltip", () => {
    it("shows a short tooltip for terms without forcing a popover click", async () => {
        const user = userEvent.setup();
        wrap(
            <HelpTooltip termId="precision" variant="label">
                Precision
            </HelpTooltip>
        );
        expect(screen.getByText("Precision")).toBeInTheDocument();
        await user.hover(screen.getByRole("button", { name: /About Precision/i }));
        expect(await screen.findByRole("tooltip")).toHaveTextContent(/true positives|predicted/i);
    });

    it("opens a popover when detail exists", async () => {
        const user = userEvent.setup();
        wrap(<HelpFieldLabel termId="cohens_kappa">Cohen's κ</HelpFieldLabel>);
        await user.click(screen.getByRole("button", { name: /About Cohen/i }));
        expect(await screen.findByText(/two-rater designs/i)).toBeInTheDocument();
    });
});
