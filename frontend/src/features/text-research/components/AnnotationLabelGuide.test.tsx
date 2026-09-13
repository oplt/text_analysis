import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { AnnotationLabelGuide } from "./AnnotationLabelGuide";
import type { AnnotationLabel } from "../types";

const labels: AnnotationLabel[] = [
    {
        id: "l1",
        codebook_id: "cb1",
        name: "Climate",
        description: "Mentions climate policy",
        inclusion_criteria: "Explicit climate framing",
        exclusion_criteria: "Weather only",
        positive_examples: ["net zero"],
        negative_examples: ["rainy day"],
        is_placeholder: false,
        created_at: "2026-01-01T00:00:00Z",
    },
];

describe("AnnotationLabelGuide", () => {
    it("renders expandable codebook guidance without dumping every field open", () => {
        render(
            <ThemeProvider theme={createTheme()}>
                <AnnotationLabelGuide
                    labels={labels}
                    codebookName="Demo"
                    codebookVersion="1.0"
                    frozen
                />
            </ThemeProvider>
        );
        expect(screen.getByText(/Codebook Demo/)).toBeInTheDocument();
        expect(screen.getByText("Climate")).toBeInTheDocument();
        expect(screen.queryByText("Explicit climate framing")).not.toBeInTheDocument();
    });
});
