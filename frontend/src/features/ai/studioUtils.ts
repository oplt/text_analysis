import { formatCurrency } from "../../utils/formatters";
import { z } from "zod";

export function parseJsonObject(value: string, fallback: Record<string, unknown> = {}) { if (!value.trim()) return fallback; const parsed = JSON.parse(value); if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("JSON payload must be an object."); return parsed as Record<string, unknown>; }
export function formatCostMicros(micros: number) { return formatCurrency(micros / 10000, "USD"); }

const jsonObjectText = z.string().superRefine((value, context) => {
    try {
        parseJsonObject(value);
    } catch (error) {
        context.addIssue({ code: "custom", message: error instanceof Error ? error.message : "Invalid JSON object." });
    }
});

const numericText = (label: string, minimum: number, maximum: number) =>
    z.string().refine((value) => {
        const number = Number(value);
        return Number.isFinite(number) && number >= minimum && number <= maximum;
    }, `${label} must be between ${minimum} and ${maximum}.`);

export const aiRunDraftSchema = z.object({
    prompt_template_key: z.string().min(2, "Select a prompt template."),
    variables_json: jsonObjectText,
    retrieval_query: z.string().max(4000),
    top_k: numericText("Top K", 1, 20),
    review_required: z.boolean(),
});

export const promptVersionDraftSchema = z.object({
    provider_key: z.string().min(2),
    model_name: z.string().trim().min(2, "Model name is required."),
    system_prompt: z.string(),
    user_prompt_template: z.string().trim().min(1, "User prompt template is required."),
    variable_names: z.string(),
    response_format: z.enum(["text", "json"]),
    temperature: numericText("Temperature", 0, 2),
    rollout_percentage: numericText("Rollout", 0, 100),
    is_published: z.boolean(),
    input_cost_per_million: numericText("Input cost", 0, Number.MAX_SAFE_INTEGER),
    output_cost_per_million: numericText("Output cost", 0, Number.MAX_SAFE_INTEGER),
});

export const datasetCaseDraftSchema = z.object({
    input_variables_json: jsonObjectText,
    retrieval_query: z.string().max(4000),
    expected_chunk_ids: z.string(),
    expected_output_text: z.string(),
    expected_output_json: z.string().superRefine((value, context) => {
        if (!value.trim()) return;
        try {
            parseJsonObject(value);
        } catch (error) {
            context.addIssue({ code: "custom", message: error instanceof Error ? error.message : "Invalid JSON object." });
        }
    }),
    notes: z.string(),
});

export function firstSchemaError(result: { success: false; error: z.ZodError }) {
    return result.error.issues[0]?.message ?? "Invalid form values.";
}
