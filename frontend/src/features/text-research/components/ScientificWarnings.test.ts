import { describe, expect, it } from "vitest";

import { collectScientificWarnings } from "../components/scientificWarnings";

describe("collectScientificWarnings", () => {
    it("dedupes diagnostics across result payloads", () => {
        const warnings = collectScientificWarnings(
            { scientific_warnings: ["Warning: Only 14 overlapping annotations"] },
            { diagnostics: ["Warning: Only 14 overlapping annotations", "Warning: high missingness"] },
            { warnings: ["Warning: sparsity is high"] }
        );
        expect(warnings).toEqual([
            "Warning: Only 14 overlapping annotations",
            "Warning: high missingness",
            "Warning: sparsity is high",
        ]);
    });
});
