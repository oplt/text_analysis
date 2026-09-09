import type { AnnotationBlindPolicy, AnnotationQueueCampaignSummary } from "./types";

type BlindPolicyInput = {
    blindPolicy?: AnnotationBlindPolicy | null;
    campaign?: AnnotationQueueCampaignSummary | null;
};

type PredictionPolicyInput = BlindPolicyInput & {
    selectedModelId: string | null;
};

export function shouldFetchPredictions({
    selectedModelId,
    blindPolicy,
    campaign,
}: PredictionPolicyInput): boolean {
    if (!selectedModelId) return false;

    const selectedBlind =
        blindPolicy?.hide_model_predictions ?? campaign?.blind_mode ?? false;
    const allowAiAssist =
        blindPolicy?.ai_assistance_enabled === true ||
        campaign?.ai_assistance_enabled === true;

    return allowAiAssist && !selectedBlind;
}

export function shouldFetchPeerAnnotations({
    blindPolicy,
    campaign,
}: BlindPolicyInput): boolean {
    if (blindPolicy?.hide_peer_annotations === true) return false;
    if (blindPolicy?.hide_peer_annotations === false) return true;
    return !(campaign?.blind_mode ?? false);
}

export function shouldFetchAdjudications({
    blindPolicy,
    campaign,
}: BlindPolicyInput): boolean {
    if (blindPolicy?.hide_adjudications === true) return false;
    if (blindPolicy?.hide_adjudications === false) return true;
    return !(campaign?.blind_mode ?? false);
}

export function isBlindReliabilityCoding(
    blindPolicy?: AnnotationBlindPolicy | null,
    campaign?: AnnotationQueueCampaignSummary | null
): boolean {
    return blindPolicy?.hide_model_predictions ?? campaign?.blind_mode ?? false;
}
