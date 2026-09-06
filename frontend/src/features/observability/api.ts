import { apiFetch } from "../../api/client";
import type { ObservabilityLinks, ObservabilityStatus } from "./types";

export function getObservabilityLinks(): Promise<ObservabilityLinks> {
    return apiFetch("/observability/links");
}

export function getObservabilityStatus(): Promise<ObservabilityStatus> {
    return apiFetch("/observability/status");
}
