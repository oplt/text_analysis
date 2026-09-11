import { useMemo, useRef, useState } from "react";
import {
    Box,
    Button,
    Checkbox,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TablePagination,
    TableRow,
    TextField,
    Typography,
} from "@mui/material";
import {
    Add as AddIcon,
    Description as DocsIcon,
    PlayArrow as SegmentIcon,
    Science as SeedIcon,
    UploadFile as CsvIcon,
} from "@mui/icons-material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import { useDebounce } from "../../../hooks/useDebounce";
import {
    bulkUpdateDocumentMetadata,
    createCorpus,
    getDashboardSummary,
    importDocumentMetadataCsv,
    listDocuments,
    seedDemoCorpus,
} from "../../../api/textResearch";
import { EmptyState } from "../../../components/ui/EmptyState";
import { PageTabs } from "../../../components/ui/PageTabs";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { queryKeys } from "../../../config/queryKeys";
import { getQueryErrorMessage } from "../../../utils/queryErrors";
import { CorpusDocumentDrawer } from "../components/CorpusDocumentDrawer";
import { CorpusDocumentFilters } from "../components/CorpusDocumentFilters";
import {
    DEFAULT_DOCUMENT_FILTERS,
    type CorpusDocumentFiltersState,
} from "../components/corpusDocumentFiltersModel";
import { CorpusUploadPanel } from "../components/CorpusUploadPanel";
import { NoCorpusEmptyState } from "../components/ResearchShared";
import { useResearchContext } from "../hooks/useResearchContext";
import type { CorpusDocument } from "../types";

const PAGE_SIZE = 25;

type CorpusTab = "management" | "prepare" | "documents";

const CORPUS_TAB_ITEMS: Array<{ value: CorpusTab; label: string }> = [
    { value: "management", label: "Corpus management" },
    { value: "prepare", label: "Prepare corpus" },
    { value: "documents", label: "Documents" },
];

export default function CorpusView() {
    const ctx = useResearchContext();
    const client = useQueryClient();
    const navigate = useNavigate();
    const { showToast } = useSnackbar();
    const csvInputRef = useRef<HTMLInputElement>(null);

    const [tab, setTab] = useState<CorpusTab>("management");
    const [createOpen, setCreateOpen] = useState(false);
    const [corpusName, setCorpusName] = useState("");
    const [corpusDescription, setCorpusDescription] = useState("");
    const [filters, setFilters] = useState<CorpusDocumentFiltersState>(DEFAULT_DOCUMENT_FILTERS);
    const [page, setPage] = useState(0);
    const [selectedIds, setSelectedIds] = useState<string[]>([]);
    const [selectedDoc, setSelectedDoc] = useState<CorpusDocument | null>(null);
    const [bulkOrg, setBulkOrg] = useState("");
    const [bulkYear, setBulkYear] = useState("");
    const [bulkLanguage, setBulkLanguage] = useState("");
    const debouncedSearch = useDebounce(filters.search.trim(), 300);

    const listParams = useMemo(
        () => ({
            limit: PAGE_SIZE,
            offset: page * PAGE_SIZE,
            organization: filters.organization.trim() || undefined,
            publication_year: filters.publication_year.trim()
                ? Number(filters.publication_year)
                : undefined,
            region: filters.region.trim() || undefined,
            cultural_sphere: filters.cultural_sphere.trim() || undefined,
            language: filters.language.trim() || undefined,
            search: debouncedSearch || undefined,
            sort_by: filters.sort_by,
            sort_dir: filters.sort_dir,
        }),
        [debouncedSearch, filters, page]
    );

    const documentsQuery = useQuery({
        queryKey: queryKeys.textResearch.documents(ctx.selectedCorpusId, listParams),
        queryFn: ({ signal }) => listDocuments(ctx.selectedCorpusId, listParams, signal),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const dashboardQuery = useQuery({
        queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
        queryFn: () => getDashboardSummary(ctx.selectedCorpusId),
        enabled: Boolean(ctx.selectedCorpusId),
    });

    const invalidateCorpus = () => {
        void client.invalidateQueries({
            queryKey: ["text-research", "corpus", ctx.selectedCorpusId, "documents"],
        });
        void client.invalidateQueries({
            queryKey: queryKeys.textResearch.dashboard(ctx.selectedCorpusId),
        });
    };

    const seedMutation = useMutation({
        mutationFn: () => seedDemoCorpus(ctx.projectId),
        onSuccess: (corpus) => {
            void client.invalidateQueries({ queryKey: queryKeys.textResearch.corpora(ctx.projectId) });
            ctx.setSelectedCorpusId(corpus.id);
            showToast({ message: "Demo corpus seeded.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to seed demo corpus."),
                severity: "error",
            }),
    });

    const createMutation = useMutation({
        mutationFn: () =>
            createCorpus(ctx.projectId, {
                name: corpusName.trim(),
                description: corpusDescription.trim() || undefined,
            }),
        onSuccess: (corpus) => {
            void client.invalidateQueries({ queryKey: queryKeys.textResearch.corpora(ctx.projectId) });
            ctx.setSelectedCorpusId(corpus.id);
            setCreateOpen(false);
            setCorpusName("");
            setCorpusDescription("");
            showToast({ message: "Corpus created.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Failed to create corpus."),
                severity: "error",
            }),
    });

    const bulkMutation = useMutation({
        mutationFn: () => {
            const fields: Record<string, unknown> = {};
            if (bulkOrg.trim()) fields.organization = bulkOrg.trim();
            if (bulkYear.trim()) fields.publication_year = Number(bulkYear);
            if (bulkLanguage.trim()) fields.language = bulkLanguage.trim();
            if (!Object.keys(fields).length) {
                throw new Error("Enter at least one bulk metadata field.");
            }
            return bulkUpdateDocumentMetadata(ctx.selectedCorpusId, {
                document_ids: selectedIds,
                fields,
            });
        },
        onSuccess: () => {
            invalidateCorpus();
            setSelectedIds([]);
            showToast({ message: "Bulk metadata updated.", severity: "success" });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "Bulk update failed."),
                severity: "error",
            }),
    });

    const csvImportMutation = useMutation({
        mutationFn: (file: File) => importDocumentMetadataCsv(ctx.selectedCorpusId, file),
        onSuccess: (result) => {
            invalidateCorpus();
            showToast({
                message: `CSV import: ${result.updated} updated, ${result.errors.length} error(s).`,
                severity: result.errors.length ? "warning" : "success",
            });
        },
        onError: (error) =>
            showToast({
                message: getQueryErrorMessage(error, "CSV import failed."),
                severity: "error",
            }),
    });

    if (!ctx.corporaLoading && ctx.corpora.length === 0) {
        return <NoCorpusEmptyState />;
    }

    const documentCount = documentsQuery.data?.total ?? 0;
    const unitCount = dashboardQuery.data?.text_unit_counts?.[ctx.unitType] ?? 0;
    const needsSegmentation = Boolean(ctx.selectedCorpusId) && documentCount > 0 && unitCount === 0;
    const pageItems = documentsQuery.data?.items ?? [];

    function toggleSelected(id: string) {
        setSelectedIds((current) =>
            current.includes(id) ? current.filter((item) => item !== id) : [...current, id]
        );
    }

    function toggleAllOnPage() {
        const ids = pageItems.map((doc) => doc.id);
        const allSelected = ids.every((id) => selectedIds.includes(id));
        setSelectedIds((current) =>
            allSelected
                ? current.filter((id) => !ids.includes(id))
                : Array.from(new Set([...current, ...ids]))
        );
    }

    return (
        <Stack spacing={2}>
            <PageTabs
                value={tab}
                onChange={setTab}
                tabs={CORPUS_TAB_ITEMS}
                ariaLabel="Corpus workflow"
            />

            {tab === "management" ? (
                <SectionCard
                    title="Corpus management"
                    description="Upload source documents through RAG ingestion, then manage research metadata."
                    action={
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                            <Button
                                variant="outlined"
                                startIcon={<SeedIcon />}
                                onClick={() => seedMutation.mutate()}
                                disabled={seedMutation.isPending}
                            >
                                Seed demo
                            </Button>
                            <Button
                                variant="contained"
                                startIcon={<AddIcon />}
                                onClick={() => setCreateOpen(true)}
                            >
                                New corpus
                            </Button>
                        </Stack>
                    }
                >
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Active corpus: {ctx.selectedCorpus?.name ?? "None selected"}
                    </Typography>
                    <CorpusUploadPanel
                        corpusId={ctx.selectedCorpusId}
                        disabled={!ctx.selectedCorpusId}
                        onComplete={invalidateCorpus}
                    />
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1} sx={{ mt: 2 }}>
                        <Button
                            variant="contained"
                            color="secondary"
                            startIcon={<SegmentIcon />}
                            onClick={() => navigate(`/research/${ctx.projectId}/prepare`)}
                            disabled={!ctx.selectedCorpusId || documentCount === 0}
                        >
                            Prepare / Segment into {ctx.unitType}s
                        </Button>
                        <Button
                            variant="outlined"
                            startIcon={<CsvIcon />}
                            disabled={!ctx.selectedCorpusId || csvImportMutation.isPending}
                            onClick={() => csvInputRef.current?.click()}
                        >
                            Import metadata CSV
                        </Button>
                        <Box
                            component="input"
                            ref={csvInputRef}
                            type="file"
                            accept=".csv,text/csv"
                            hidden
                            onChange={(event) => {
                                const file = event.target.files?.[0];
                                if (file) csvImportMutation.mutate(file);
                                event.target.value = "";
                            }}
                        />
                    </Stack>
                </SectionCard>
            ) : null}

            {tab === "prepare" ? (
                <SectionCard
                    title="Prepare corpus"
                    description={
                        needsSegmentation
                            ? "Documents are ready — create research text units next."
                            : "Segment or re-prepare this corpus into research text units."
                    }
                >
                    {needsSegmentation ? (
                        <EmptyState
                            icon={<SegmentIcon fontSize="large" />}
                            title="Corpus not segmented yet"
                            description={`${documentCount} document${documentCount === 1 ? "" : "s"} linked. Selected unit type: ${ctx.unitType}.`}
                            action={
                                <Button
                                    variant="contained"
                                    startIcon={<SegmentIcon />}
                                    onClick={() => navigate(`/research/${ctx.projectId}/prepare`)}
                                >
                                    Open preparation workspace
                                </Button>
                            }
                        />
                    ) : (
                        <EmptyState
                            icon={<SegmentIcon fontSize="large" />}
                            title={
                                !ctx.selectedCorpusId
                                    ? "Select a corpus"
                                    : documentCount === 0
                                      ? "Upload documents first"
                                      : "Corpus already segmented"
                            }
                            description={
                                !ctx.selectedCorpusId
                                    ? "Choose a corpus in Corpus management, then prepare text units."
                                    : documentCount === 0
                                      ? "Add source documents before segmentation."
                                      : `${unitCount} ${ctx.unitType}${unitCount === 1 ? "" : "s"} ready. Open preparation to adjust preprocessing or re-segment.`
                            }
                            action={
                                <Button
                                    variant="contained"
                                    startIcon={<SegmentIcon />}
                                    onClick={() => navigate(`/research/${ctx.projectId}/prepare`)}
                                    disabled={!ctx.selectedCorpusId || documentCount === 0}
                                >
                                    Open preparation workspace
                                </Button>
                            }
                        />
                    )}
                </SectionCard>
            ) : null}

            {tab === "documents" ? (
                <SectionCard
                    title="Documents"
                    description="Server-paginated corpus documents. Select a row to edit metadata and preview source text."
                >
                    {!ctx.selectedCorpusId ? (
                        <EmptyState
                            icon={<DocsIcon fontSize="large" />}
                            title="Select a corpus"
                            description="Choose a corpus in Corpus management, or create one to list and manage documents."
                            action={
                                <Button
                                    variant="contained"
                                    startIcon={<AddIcon />}
                                    onClick={() => setCreateOpen(true)}
                                >
                                    Create empty corpus
                                </Button>
                            }
                        />
                    ) : (
                        <Box
                            sx={{
                                display: "grid",
                                gap: 2,
                                gridTemplateColumns: { xs: "1fr", md: "220px 1fr" },
                            }}
                        >
                            <CorpusDocumentFilters
                                value={filters}
                                onChange={(next) => {
                                    setPage(0);
                                    setFilters(next);
                                }}
                            />
                            <Box>
                                {selectedIds.length > 0 ? (
                                    <Stack
                                        direction={{ xs: "column", sm: "row" }}
                                        spacing={1}
                                        sx={{ mb: 1.5 }}
                                        alignItems={{ sm: "center" }}
                                    >
                                        <Typography variant="body2">{selectedIds.length} selected</Typography>
                                        <TextField
                                            size="small"
                                            label="Bulk organization"
                                            value={bulkOrg}
                                            onChange={(e) => setBulkOrg(e.target.value)}
                                        />
                                        <TextField
                                            size="small"
                                            label="Bulk year"
                                            value={bulkYear}
                                            onChange={(e) => setBulkYear(e.target.value)}
                                            sx={{ width: 110 }}
                                        />
                                        <TextField
                                            size="small"
                                            label="Bulk language"
                                            value={bulkLanguage}
                                            onChange={(e) => setBulkLanguage(e.target.value)}
                                            sx={{ width: 130 }}
                                        />
                                        <Button
                                            size="small"
                                            variant="outlined"
                                            onClick={() => bulkMutation.mutate()}
                                            disabled={bulkMutation.isPending}
                                        >
                                            Apply bulk update
                                        </Button>
                                    </Stack>
                                ) : null}

                                <QueryBoundary
                                    isLoading={documentsQuery.isLoading}
                                    isError={documentsQuery.isError}
                                    error={documentsQuery.error}
                                    onRetry={() => void documentsQuery.refetch()}
                                >
                                    {pageItems.length ? (
                                        <>
                                            <Box sx={{ overflowX: "auto" }}>
                                                <Table size="small">
                                                    <TableHead>
                                                        <TableRow>
                                                            <TableCell padding="checkbox">
                                                                <Checkbox
                                                                    size="small"
                                                                    checked={
                                                                        pageItems.length > 0 &&
                                                                        pageItems.every((doc) =>
                                                                            selectedIds.includes(doc.id)
                                                                        )
                                                                    }
                                                                    indeterminate={
                                                                        pageItems.some((doc) =>
                                                                            selectedIds.includes(doc.id)
                                                                        ) &&
                                                                        !pageItems.every((doc) =>
                                                                            selectedIds.includes(doc.id)
                                                                        )
                                                                    }
                                                                    onChange={toggleAllOnPage}
                                                                />
                                                            </TableCell>
                                                            <TableCell>Title</TableCell>
                                                            <TableCell>Organization</TableCell>
                                                            <TableCell>Year</TableCell>
                                                            <TableCell>Country</TableCell>
                                                            <TableCell>Language</TableCell>
                                                        </TableRow>
                                                    </TableHead>
                                                    <TableBody>
                                                        {pageItems.map((doc) => (
                                                            <TableRow
                                                                key={doc.id}
                                                                hover
                                                                selected={selectedDoc?.id === doc.id}
                                                                sx={{ cursor: "pointer" }}
                                                                onClick={() => setSelectedDoc(doc)}
                                                            >
                                                                <TableCell
                                                                    padding="checkbox"
                                                                    onClick={(event) =>
                                                                        event.stopPropagation()
                                                                    }
                                                                >
                                                                    <Checkbox
                                                                        size="small"
                                                                        checked={selectedIds.includes(doc.id)}
                                                                        onChange={() => toggleSelected(doc.id)}
                                                                    />
                                                                </TableCell>
                                                                <TableCell>
                                                                    {doc.title ?? doc.id.slice(0, 8)}
                                                                </TableCell>
                                                                <TableCell>{doc.organization ?? "—"}</TableCell>
                                                                <TableCell>
                                                                    {doc.publication_year ?? "—"}
                                                                </TableCell>
                                                                <TableCell>{doc.country ?? "—"}</TableCell>
                                                                <TableCell>{doc.language ?? "—"}</TableCell>
                                                            </TableRow>
                                                        ))}
                                                    </TableBody>
                                                </Table>
                                            </Box>
                                            <TablePagination
                                                component="div"
                                                count={documentsQuery.data?.total ?? 0}
                                                page={page}
                                                onPageChange={(_, next) => setPage(next)}
                                                rowsPerPage={PAGE_SIZE}
                                                rowsPerPageOptions={[PAGE_SIZE]}
                                            />
                                        </>
                                    ) : (
                                        <EmptyState
                                            icon={<DocsIcon fontSize="large" />}
                                            title="No documents yet"
                                            description="Upload source files in Corpus management, import a metadata CSV, or load the synthetic demo."
                                        />
                                    )}
                                </QueryBoundary>
                            </Box>
                        </Box>
                    )}
                </SectionCard>
            ) : null}

            <CorpusDocumentDrawer
                open={Boolean(selectedDoc)}
                document={selectedDoc}
                corpusId={ctx.selectedCorpusId}
                onClose={() => setSelectedDoc(null)}
            />

            <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle>Create corpus</DialogTitle>
                <DialogContent>
                    <Stack spacing={2} sx={{ mt: 1 }}>
                        <TextField
                            label="Name"
                            value={corpusName}
                            onChange={(e) => setCorpusName(e.target.value)}
                            fullWidth
                            required
                        />
                        <TextField
                            label="Description"
                            value={corpusDescription}
                            onChange={(e) => setCorpusDescription(e.target.value)}
                            fullWidth
                            multiline
                            minRows={2}
                        />
                    </Stack>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
                    <Button
                        variant="contained"
                        onClick={() => createMutation.mutate()}
                        disabled={!corpusName.trim() || createMutation.isPending}
                    >
                        Create
                    </Button>
                </DialogActions>
            </Dialog>
        </Stack>
    );
}
