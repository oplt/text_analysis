export type SharedTopicFilters = {
    organization: string;
    language: string;
    region: string;
    culturalSphere: string;
    publicationYearMin: string;
    publicationYearMax: string;
};

export const DEFAULT_TOPIC_FILTERS: SharedTopicFilters = {
    organization: "",
    language: "",
    region: "",
    culturalSphere: "",
    publicationYearMin: "",
    publicationYearMax: "",
};

export function topicFilterPayload(filters: SharedTopicFilters) {
    const optionalText = (value: string): string | undefined => {
        const trimmed = value.trim();
        return trimmed ? trimmed : undefined;
    };

    return {
        organization: optionalText(filters.organization),
        language: optionalText(filters.language),
        region: optionalText(filters.region),
        cultural_sphere: optionalText(filters.culturalSphere),
        publication_year_min: filters.publicationYearMin
            ? Number(filters.publicationYearMin)
            : undefined,
        publication_year_max: filters.publicationYearMax
            ? Number(filters.publicationYearMax)
            : undefined,
    };
}
