export type CorpusDocumentFiltersState = {
    search: string;
    organization: string;
    publication_year: string;
    region: string;
    cultural_sphere: string;
    language: string;
    sort_by: string;
    sort_dir: "asc" | "desc";
};

export const DEFAULT_DOCUMENT_FILTERS: CorpusDocumentFiltersState = {
    search: "",
    organization: "",
    publication_year: "",
    region: "",
    cultural_sphere: "",
    language: "",
    sort_by: "created_at",
    sort_dir: "asc",
};
