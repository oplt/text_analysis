import { Chip, MenuItem, Stack, TextField } from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { getCorpusMetadataFacets } from "../../../api/textResearch";
import { FormGrid } from "../../../components/ui/FormGrid";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";

const FIELDS = [
    "organization",
    "organization_type",
    "publication_year",
    "country",
    "region",
    "cultural_sphere",
    "language",
    "publication_type",
] as const;

export function MetadataFilterBar({
    corpusId,
    value,
    onChange,
}: {
    corpusId: string;
    value: Record<string, string>;
    onChange: (value: Record<string, string>) => void;
}) {
    const facets = useQuery({
        queryKey: queryKeys.textResearch.metadataFacets(corpusId),
        queryFn: ({ signal }) => getCorpusMetadataFacets(corpusId, signal),
        enabled: Boolean(corpusId),
        staleTime: QUERY_STALE_TIMES.researchMetadataFacets,
    });
    const set = (field: string, next: string) => {
        const copy = { ...value };
        if (next) copy[field] = next;
        else delete copy[field];
        onChange(copy);
    };

    return (
        <Stack spacing={1}>
            <FormGrid columns="4">
                {FIELDS.map((field) => (
                    <TextField
                        key={field}
                        select
                        size="small"
                        fullWidth
                        label={field.replace(/_/g, " ")}
                        value={value[field] ?? ""}
                        onChange={(event) => set(field, event.target.value)}
                    >
                        <MenuItem value="">Any</MenuItem>
                        {(facets.data?.[field] ?? []).map((item) => (
                            <MenuItem key={item.value} value={item.value}>
                                {item.value} ({item.count})
                            </MenuItem>
                        ))}
                    </TextField>
                ))}
            </FormGrid>
            <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                {Object.entries(value).map(([field, selected]) => (
                    <Chip
                        key={field}
                        label={`${field.replace(/_/g, " ")}: ${selected}`}
                        onDelete={() => set(field, "")}
                    />
                ))}
                {Object.keys(value).length ? (
                    <Chip label="Reset filters" onClick={() => onChange({})} />
                ) : null}
            </Stack>
        </Stack>
    );
}
