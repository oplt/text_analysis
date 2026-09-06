import {
    Button,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    Stack,
    TextField,
    Typography,
} from "@mui/material";

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

type CorpusDocumentFiltersProps = {
    value: CorpusDocumentFiltersState;
    onChange: (next: CorpusDocumentFiltersState) => void;
};

export function CorpusDocumentFilters({ value, onChange }: CorpusDocumentFiltersProps) {
    function patch(partial: Partial<CorpusDocumentFiltersState>) {
        onChange({ ...value, ...partial });
    }

    return (
        <Stack spacing={1.5}>
            <Typography variant="subtitle2">Filters</Typography>
            <TextField
                size="small"
                label="Search"
                value={value.search}
                onChange={(e) => patch({ search: e.target.value })}
                placeholder="Title, org, country"
            />
            <TextField
                size="small"
                label="Organization"
                value={value.organization}
                onChange={(e) => patch({ organization: e.target.value })}
            />
            <TextField
                size="small"
                label="Publication year"
                value={value.publication_year}
                onChange={(e) => patch({ publication_year: e.target.value })}
            />
            <TextField
                size="small"
                label="Region"
                value={value.region}
                onChange={(e) => patch({ region: e.target.value })}
            />
            <TextField
                size="small"
                label="Cultural sphere"
                value={value.cultural_sphere}
                onChange={(e) => patch({ cultural_sphere: e.target.value })}
            />
            <TextField
                size="small"
                label="Language"
                value={value.language}
                onChange={(e) => patch({ language: e.target.value })}
            />
            <FormControl size="small">
                <InputLabel>Sort by</InputLabel>
                <Select
                    label="Sort by"
                    value={value.sort_by}
                    onChange={(e) => patch({ sort_by: e.target.value })}
                >
                    <MenuItem value="created_at">Created</MenuItem>
                    <MenuItem value="title">Title</MenuItem>
                    <MenuItem value="organization">Organization</MenuItem>
                    <MenuItem value="publication_year">Year</MenuItem>
                    <MenuItem value="country">Country</MenuItem>
                    <MenuItem value="language">Language</MenuItem>
                </Select>
            </FormControl>
            <FormControl size="small">
                <InputLabel>Direction</InputLabel>
                <Select
                    label="Direction"
                    value={value.sort_dir}
                    onChange={(e) => patch({ sort_dir: e.target.value as "asc" | "desc" })}
                >
                    <MenuItem value="asc">Ascending</MenuItem>
                    <MenuItem value="desc">Descending</MenuItem>
                </Select>
            </FormControl>
            <Button size="small" onClick={() => onChange(DEFAULT_DOCUMENT_FILTERS)}>
                Clear filters
            </Button>
        </Stack>
    );
}
