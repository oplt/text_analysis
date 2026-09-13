import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
    Button,
    Stack,
    Typography,
    useMediaQuery,
    useTheme,
} from "@mui/material";
import {
    Add as AddIcon,
    Description as DocsIcon,
} from "@mui/icons-material";
import { ContextInspector } from "../../../components/ui/ContextInspector";
import { DataTable, type DataTableColumn } from "../../../components/ui/DataTable";
import type { DataTableDensity } from "../../../components/ui/dataTableTokens";
import { EmptyState } from "../../../components/ui/EmptyState";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { SectionCard } from "../../../components/ui/SectionCard";
import { WorkspaceSplit } from "../../../components/ui/WorkspaceSplit";
import { CorpusDocumentDrawer } from "./CorpusDocumentDrawer";
import { CorpusDocumentFilters } from "./CorpusDocumentFilters";
import { CorpusDocumentInspector } from "./CorpusDocumentInspector";
import type { CorpusDocumentFiltersState } from "./corpusDocumentFiltersModel";
import type { CorpusDocument, UnitType } from "../types";

export type DocumentTableDensity = DataTableDensity;

type CorpusDocumentsWorkspaceProps = {
    corpusId: string | null;
    filters: CorpusDocumentFiltersState;
    onFiltersChange: (next: CorpusDocumentFiltersState) => void;
    page: number;
    onPageChange: (page: number) => void;
    pageSize: number;
    documents: CorpusDocument[];
    total: number;
    isLoading: boolean;
    isError: boolean;
    error: unknown;
    onRetry: () => void;
    selectedIds: string[];
    onToggleSelected: (id: string) => void;
    onToggleAllOnPage: () => void;
    selectedDoc: CorpusDocument | null;
    onSelectDoc: (doc: CorpusDocument | null) => void;
    onOpenCreateCorpus: () => void;
    onBulkEdit: () => void;
    unitType: UnitType;
    unitCountForType: number;
};

/**
 * Zotero-style master-detail document browser for a research corpus.
 */
export function CorpusDocumentsWorkspace({
    corpusId,
    filters,
    onFiltersChange,
    page,
    onPageChange,
    pageSize,
    documents,
    total,
    isLoading,
    isError,
    error,
    onRetry,
    selectedIds,
    onToggleSelected,
    onToggleAllOnPage,
    selectedDoc,
    onSelectDoc,
    onOpenCreateCorpus,
    onBulkEdit,
    unitType,
    unitCountForType,
}: CorpusDocumentsWorkspaceProps) {
    const theme = useTheme();
    const isMdUp = useMediaQuery(theme.breakpoints.up("md"));
    const [density, setDensity] = useState<DocumentTableDensity>("compact");
    const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
    const tableRef = useRef<HTMLDivElement>(null);

    const selectedIndex = useMemo(
        () => (selectedDoc ? documents.findIndex((doc) => doc.id === selectedDoc.id) : -1),
        [documents, selectedDoc]
    );

    const selectByIndex = useCallback(
        (index: number) => {
            if (index < 0 || index >= documents.length) return;
            const doc = documents[index];
            onSelectDoc(doc);
            if (!isMdUp) setMobileDrawerOpen(true);
        },
        [documents, isMdUp, onSelectDoc]
    );

    useEffect(() => {
        if (isMdUp) setMobileDrawerOpen(false);
    }, [isMdUp]);

    function handleRowActivate(doc: CorpusDocument) {
        onSelectDoc(doc);
        if (!isMdUp) setMobileDrawerOpen(true);
    }

    function handleTableKeyDown(event: React.KeyboardEvent) {
        if (!documents.length) return;
        if (event.key === "ArrowDown") {
            event.preventDefault();
            selectByIndex(selectedIndex < 0 ? 0 : Math.min(documents.length - 1, selectedIndex + 1));
        } else if (event.key === "ArrowUp") {
            event.preventDefault();
            selectByIndex(selectedIndex < 0 ? 0 : Math.max(0, selectedIndex - 1));
        } else if (event.key === "Home") {
            event.preventDefault();
            selectByIndex(0);
        } else if (event.key === "End") {
            event.preventDefault();
            selectByIndex(documents.length - 1);
        } else if (event.key === "Enter" && selectedDoc) {
            event.preventDefault();
            if (!isMdUp) setMobileDrawerOpen(true);
        } else if (event.key === "Escape") {
            event.preventDefault();
            onSelectDoc(null);
            setMobileDrawerOpen(false);
        } else if (event.key === " " && selectedDoc) {
            event.preventDefault();
            onToggleSelected(selectedDoc.id);
        }
    }

    const columns = useMemo<Array<DataTableColumn<CorpusDocument>>>(
        () => [
            {
                id: "title",
                label: "Title",
                sticky: "left",
                sortable: true,
                truncate: 56,
                minWidth: 160,
                getSortValue: (doc) => doc.title ?? doc.id,
                render: (doc) => doc.title ?? doc.id.slice(0, 8),
            },
            {
                id: "organization",
                label: "Organization",
                sortable: true,
                truncate: true,
                getSortValue: (doc) => doc.organization ?? "",
                render: (doc) => doc.organization ?? "—",
            },
            {
                id: "year",
                label: "Year",
                sortable: true,
                getSortValue: (doc) => doc.publication_year ?? 0,
                render: (doc) => doc.publication_year ?? "—",
            },
            {
                id: "country",
                label: "Country",
                sortable: true,
                hideable: true,
                getSortValue: (doc) => doc.country ?? "",
                render: (doc) => doc.country ?? "—",
            },
            {
                id: "language",
                label: "Language",
                sortable: true,
                hideable: true,
                getSortValue: (doc) => doc.language ?? "",
                render: (doc) => doc.language ?? "—",
            },
        ],
        []
    );

    if (!corpusId) {
        return (
            <SectionCard title="Documents" description="Browse and inspect corpus documents.">
                <EmptyState
                    icon={<DocsIcon fontSize="large" />}
                    title="Select a corpus"
                    description="Choose a corpus under Import, or create one to list and manage documents."
                    action={
                        <Button variant="contained" startIcon={<AddIcon />} onClick={onOpenCreateCorpus}>
                            Create empty corpus
                        </Button>
                    }
                />
            </SectionCard>
        );
    }

    const table = (
        <SectionCard
            title="Documents"
            description="Search, filter, and select documents. Details open in the inspector."
        >
            <Stack spacing={1.5}>
                <CorpusDocumentFilters
                    variant="toolbar"
                    value={filters}
                    onChange={(next) => {
                        onPageChange(0);
                        onFiltersChange(next);
                    }}
                />

                {selectedIds.length > 0 ? (
                    <Stack
                        direction={{ xs: "column", sm: "row" }}
                        spacing={1}
                        alignItems={{ sm: "center" }}
                    >
                        <Typography variant="body2">{selectedIds.length} selected</Typography>
                        <Button size="small" variant="outlined" onClick={onBulkEdit}>
                            Bulk edit metadata
                        </Button>
                    </Stack>
                ) : null}

                <QueryBoundary
                    isLoading={isLoading}
                    isError={isError}
                    error={error}
                    onRetry={onRetry}
                >
                    <DataTable
                        ariaLabel="Corpus documents"
                        columns={columns}
                        rows={documents}
                        getRowId={(doc) => doc.id}
                        density={density}
                        onDensityChange={setDensity}
                        showDensityToggle
                        showColumnVisibility
                        stickyHeader
                        stickyFirstColumn
                        loading={false}
                        emptyIcon={<DocsIcon fontSize="large" />}
                        emptyTitle="No documents yet"
                        emptyDescription="Upload source files under Import, import a metadata CSV, or load the synthetic demo."
                        selectedIds={selectedIds}
                        onToggleSelected={onToggleSelected}
                        onToggleAllOnPage={onToggleAllOnPage}
                        clientSort
                        page={page}
                        pageSize={pageSize}
                        totalCount={total}
                        onPageChange={onPageChange}
                        onRowClick={handleRowActivate}
                        selectedRowId={selectedDoc?.id ?? null}
                        onKeyDown={handleTableKeyDown}
                        containerRef={tableRef}
                        footerStart={
                            <Typography variant="caption" color="text.secondary">
                                ↑↓ navigate · Space select · Esc clear
                                {!isMdUp ? " · Enter open details" : ""}
                            </Typography>
                        }
                    />
                </QueryBoundary>
            </Stack>
        </SectionCard>
    );

    const inspector = (
        <ContextInspector
            title={selectedDoc?.title ?? "Document inspector"}
            description="Metadata, ingestion, segmentation, provenance, and source text."
            sticky
            dismissible={Boolean(selectedDoc)}
            onDismiss={() => onSelectDoc(null)}
            sx={{ width: "100%", maxWidth: "none", minWidth: 0 }}
        >
            <CorpusDocumentInspector
                document={selectedDoc}
                corpusId={corpusId}
                unitType={unitType}
                unitCountForType={unitCountForType}
                active={Boolean(selectedDoc) && isMdUp}
                onRemoved={() => onSelectDoc(null)}
                onSaved={(updated) => onSelectDoc(updated)}
            />
        </ContextInspector>
    );

    return (
        <>
            <WorkspaceSplit main={table} side={inspector} />
            {!isMdUp ? (
                <CorpusDocumentDrawer
                    open={mobileDrawerOpen && Boolean(selectedDoc)}
                    onClose={() => {
                        setMobileDrawerOpen(false);
                        onSelectDoc(null);
                    }}
                    document={selectedDoc}
                    corpusId={corpusId}
                />
            ) : null}
        </>
    );
}
