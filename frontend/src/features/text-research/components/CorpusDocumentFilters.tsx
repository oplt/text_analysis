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
import { AdvancedSettings } from "../../../components/ui/AdvancedSettings";
import { FormGrid } from "../../../components/ui/FormGrid";
import {
    DEFAULT_DOCUMENT_FILTERS,
    type CorpusDocumentFiltersState,
} from "./corpusDocumentFiltersModel";

type CorpusDocumentFiltersProps = {
    value: CorpusDocumentFiltersState;
    onChange: (next: CorpusDocumentFiltersState) => void;
    /** Compact toolbar for the documents master list; panel for side filters. */
    variant?: "panel" | "toolbar";
};

export function CorpusDocumentFilters({
    value,
    onChange,
    variant = "panel",
}: CorpusDocumentFiltersProps) {
    function patch(partial: Partial<CorpusDocumentFiltersState>) {
        onChange({ ...value, ...partial });
    }

    const searchField = (
        <TextField
            size="small"
            label="Search"
            value={value.search}
            onChange={(e) => patch({ search: e.target.value })}
            placeholder="Title, organization, country"
            fullWidth
        />
    );

    const sortFields = (
        <>
            <FormControl size="small" sx={{ minWidth: 140 }}>
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
            <FormControl size="small" sx={{ minWidth: 120 }}>
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
        </>
    );

    const facetFields = (
        <FormGrid columns="4-4-4">
            <TextField
                size="small"
                label="Organization"
                value={value.organization}
                onChange={(e) => patch({ organization: e.target.value })}
                fullWidth
            />
            <TextField
                size="small"
                label="Publication year"
                value={value.publication_year}
                onChange={(e) => patch({ publication_year: e.target.value })}
                fullWidth
            />
            <TextField
                size="small"
                label="Language"
                value={value.language}
                onChange={(e) => patch({ language: e.target.value })}
                fullWidth
            />
            <TextField
                size="small"
                label="Region"
                value={value.region}
                onChange={(e) => patch({ region: e.target.value })}
                fullWidth
            />
            <TextField
                size="small"
                label="Cultural sphere"
                value={value.cultural_sphere}
                onChange={(e) => patch({ cultural_sphere: e.target.value })}
                fullWidth
            />
        </FormGrid>
    );

    if (variant === "toolbar") {
        return (
            <Stack spacing={1.25}>
                <FormGrid columns="8-4">
                    {searchField}
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                        {sortFields}
                    </Stack>
                </FormGrid>
                <AdvancedSettings title="Filters" description="Organization, year, language, region">
                    <Stack spacing={1.25}>
                        {facetFields}
                        <Button size="small" onClick={() => onChange(DEFAULT_DOCUMENT_FILTERS)}>
                            Clear filters
                        </Button>
                    </Stack>
                </AdvancedSettings>
            </Stack>
        );
    }

    return (
        <Stack spacing={1.5}>
            <Typography variant="subtitle2">Filters</Typography>
            {searchField}
            {facetFields}
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {sortFields}
            </Stack>
            <Button size="small" onClick={() => onChange(DEFAULT_DOCUMENT_FILTERS)}>
                Clear filters
            </Button>
        </Stack>
    );
}
