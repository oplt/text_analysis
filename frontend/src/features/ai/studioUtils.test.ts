import { describe, expect, it } from "vitest";
import {
    aiRunDraftSchema,
    datasetCaseDraftSchema,
    parseJsonObject,
    promptVersionDraftSchema,
} from "./studioUtils";

describe("AI Studio validation", () => {
    it("accepts a bounded run draft", () => {
        const result = aiRunDraftSchema.safeParse({
            prompt_template_key: "summary",
            variables_json: '{"task":"summarize"}',
            retrieval_query: "policy",
            top_k: "4",
            review_required: false,
        });
        expect(result.success).toBe(true);
    });

    it("rejects arrays and out-of-range retrieval limits", () => {
        const result = aiRunDraftSchema.safeParse({
            prompt_template_key: "summary",
            variables_json: "[]",
            retrieval_query: "",
            top_k: "21",
            review_required: false,
        });
        expect(result.success).toBe(false);
    });

    it("validates numeric prompt version constraints", () => {
        const result = promptVersionDraftSchema.safeParse({
            provider_key: "local",
            model_name: "model",
            system_prompt: "",
            user_prompt_template: "{{task}}",
            variable_names: "task",
            response_format: "text",
            temperature: "3",
            rollout_percentage: "100",
            is_published: true,
            input_cost_per_million: "0",
            output_cost_per_million: "0",
        });
        expect(result.success).toBe(false);
    });

    it("allows retrieval-only evaluation cases", () => {
        const result = datasetCaseDraftSchema.safeParse({
            input_variables_json: "{}",
            retrieval_query: "return policy",
            expected_chunk_ids: "chunk-1",
            expected_output_text: "",
            expected_output_json: "",
            notes: "",
        });
        expect(result.success).toBe(true);
        expect(parseJsonObject("{}")).toEqual({});
    });
});
