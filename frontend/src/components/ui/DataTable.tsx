import {
    Fragment,
    useMemo,
    useState,
    type KeyboardEventHandler,
    type ReactNode,
} from "react";
import {
    Box,
    Checkbox,
    Collapse,
    IconButton,
    Menu,
    MenuItem,
    Skeleton,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TablePagination,
    TableRow,
    TableSortLabel,
    ToggleButton,
    ToggleButtonGroup,
    Typography,
} from "@mui/material";
import {
    KeyboardArrowDown as ExpandIcon,
    KeyboardArrowRight as CollapseIcon,
    ViewColumn as ColumnsIcon,
} from "@mui/icons-material";
import { EmptyState } from "./EmptyState";
import { TruncatedCell } from "./TruncatedCell";
import {
    dataTableDefaults,
    dataTableDensity,
    tableSizeForDensity,
    type DataTableDensity,
} from "./dataTableTokens";
import { stickyEdgeShadow } from "./themeSurfaces";
import type { Theme } from "@mui/material/styles";

export type { DataTableDensity };

export type DataTableSort = { id: string; direction: "asc" | "desc" };

export type DataTableColumn<T> = {
    id: string;
    label: ReactNode;
    align?: "left" | "right" | "center";
    width?: number | string;
    minWidth?: number;
    /** Sticky left (first) or right (actions) column. */
    sticky?: "left" | "right";
    sortable?: boolean;
    hideable?: boolean;
    defaultHidden?: boolean;
    /** Truncate stringish render output; number = max chars. */
    truncate?: boolean | number;
    padding?: "normal" | "checkbox" | "none";
    render: (row: T) => ReactNode;
    getSortValue?: (row: T) => string | number | null | undefined;
};

export type DataTableProps<T> = {
    columns: Array<DataTableColumn<T>>;
    rows: T[];
    getRowId: (row: T, index: number) => string;
    ariaLabel: string;
    density?: DataTableDensity;
    onDensityChange?: (density: DataTableDensity) => void;
    showDensityToggle?: boolean;
    showColumnVisibility?: boolean;
    stickyHeader?: boolean;
    stickyFirstColumn?: boolean;
    maxHeight?: object | string | number;
    loading?: boolean;
    emptyIcon?: ReactNode;
    emptyTitle?: string;
    emptyDescription?: string;
    emptyAction?: ReactNode;
    /** Client-side filter: keep rows whose visible text matches (case-insensitive). */
    filterText?: string;
    toolbarStart?: ReactNode;
    toolbarEnd?: ReactNode;
    footerStart?: ReactNode;
    selectedIds?: string[];
    onToggleSelected?: (id: string) => void;
    onToggleAllOnPage?: () => void;
    sort?: DataTableSort | null;
    onSortChange?: (sort: DataTableSort | null) => void;
    /** When true and onSortChange omitted, sort is managed internally. */
    clientSort?: boolean;
    page?: number;
    pageSize?: number;
    /** Server/total count; defaults to filtered row length for client paging. */
    totalCount?: number;
    onPageChange?: (page: number) => void;
    onPageSizeChange?: (pageSize: number) => void;
    rowsPerPageOptions?: number[];
    onRowClick?: (row: T) => void;
    selectedRowId?: string | null;
    expandedRowId?: string | null;
    onExpandedRowChange?: (id: string | null) => void;
    renderExpandedRow?: (row: T) => ReactNode;
    onKeyDown?: KeyboardEventHandler<HTMLDivElement>;
    containerRef?: React.Ref<HTMLDivElement>;
};

function stickyCellSx(
    sticky: "left" | "right" | undefined,
    isHeader: boolean
): Record<string, unknown> | undefined {
    if (!sticky) return undefined;
    const z = isHeader
        ? dataTableDefaults.stickyZIndex.stickyHeaderColumn
        : dataTableDefaults.stickyZIndex.stickyColumn;
    return {
        position: "sticky",
        [sticky]: 0,
        zIndex: z,
        bgcolor: isHeader ? "inherit" : "background.paper",
        boxShadow: (theme: Theme) => stickyEdgeShadow(theme, sticky),
    };
}

function compareSortValues(
    left: string | number | null | undefined,
    right: string | number | null | undefined
): number {
    if (typeof left === "number" && typeof right === "number") return left - right;
    return String(left ?? "").localeCompare(String(right ?? ""), undefined, { numeric: true });
}

/**
 * Shared data-dense table shell: density, sticky chrome, sort, filter, selection,
 * pagination, column visibility, expansion, truncated cells, empty/loading.
 */
export function DataTable<T>({
    columns,
    rows,
    getRowId,
    ariaLabel,
    density: densityProp,
    onDensityChange,
    showDensityToggle = false,
    showColumnVisibility = false,
    stickyHeader = true,
    stickyFirstColumn = false,
    maxHeight = dataTableDefaults.maxHeight,
    loading = false,
    emptyIcon,
    emptyTitle = "No rows",
    emptyDescription = "Nothing to show yet.",
    emptyAction,
    filterText,
    toolbarStart,
    toolbarEnd,
    footerStart,
    selectedIds,
    onToggleSelected,
    onToggleAllOnPage,
    sort: sortProp,
    onSortChange,
    clientSort = false,
    page: pageProp,
    pageSize: pageSizeProp = 25,
    totalCount,
    onPageChange,
    onPageSizeChange,
    rowsPerPageOptions,
    onRowClick,
    selectedRowId,
    expandedRowId,
    onExpandedRowChange,
    renderExpandedRow,
    onKeyDown,
    containerRef,
}: DataTableProps<T>) {
    const [internalDensity, setInternalDensity] = useState<DataTableDensity>("compact");
    const density = densityProp ?? internalDensity;
    const setDensity = (next: DataTableDensity) => {
        onDensityChange?.(next);
        if (densityProp == null) setInternalDensity(next);
    };

    const [hiddenIds, setHiddenIds] = useState<Set<string>>(
        () => new Set(columns.filter((c) => c.defaultHidden).map((c) => c.id))
    );
    const [columnMenuAnchor, setColumnMenuAnchor] = useState<null | HTMLElement>(null);
    const [internalSort, setInternalSort] = useState<DataTableSort | null>(null);
    const [internalPage, setInternalPage] = useState(0);
    const [internalPageSize, setInternalPageSize] = useState(pageSizeProp);

    const sort = sortProp !== undefined ? sortProp : clientSort ? internalSort : null;
    const page = pageProp ?? internalPage;
    const pageSize = pageSizeProp ?? internalPageSize;

    const resolvedColumns = useMemo(() => {
        return columns.map((column, index) => {
            if (stickyFirstColumn && index === 0 && !column.sticky) {
                return { ...column, sticky: "left" as const };
            }
            return column;
        });
    }, [columns, stickyFirstColumn]);

    const visibleColumns = useMemo(
        () => resolvedColumns.filter((column) => !hiddenIds.has(column.id)),
        [resolvedColumns, hiddenIds]
    );

    const filteredRows = useMemo(() => {
        const q = filterText?.trim().toLowerCase();
        if (!q) return rows;
        return rows.filter((row) =>
            visibleColumns.some((column) => {
                const rendered = column.getSortValue?.(row);
                if (rendered != null && String(rendered).toLowerCase().includes(q)) return true;
                const node = column.render(row);
                if (typeof node === "string" || typeof node === "number") {
                    return String(node).toLowerCase().includes(q);
                }
                return false;
            })
        );
    }, [filterText, rows, visibleColumns]);

    const sortedRows = useMemo(() => {
        if (!sort) return filteredRows;
        const column = resolvedColumns.find((item) => item.id === sort.id);
        if (!column?.getSortValue && !column?.sortable) return filteredRows;
        return [...filteredRows].sort((a, b) => {
            const left = column.getSortValue?.(a);
            const right = column.getSortValue?.(b);
            const comparison = compareSortValues(left, right);
            return sort.direction === "asc" ? comparison : -comparison;
        });
    }, [filteredRows, resolvedColumns, sort]);

    const clientPaging = pageProp == null && totalCount == null;
    const count = totalCount ?? sortedRows.length;
    const pageCount = Math.max(1, Math.ceil(count / pageSize));
    const currentPage = Math.min(page, pageCount - 1);
    const visibleRows = clientPaging
        ? sortedRows.slice(currentPage * pageSize, (currentPage + 1) * pageSize)
        : sortedRows;

    const showSelection = Boolean(selectedIds && onToggleSelected);
    const densityTokens = dataTableDensity[density];
    const colSpan =
        visibleColumns.length + (showSelection ? 1 : 0) + (renderExpandedRow ? 1 : 0);

    function handleSort(id: string) {
        const next: DataTableSort =
            sort?.id === id
                ? { id, direction: sort.direction === "asc" ? "desc" : "asc" }
                : { id, direction: "asc" };
        if (onSortChange) onSortChange(next);
        else if (clientSort || sortProp === undefined) setInternalSort(next);
        if (pageProp == null) setInternalPage(0);
        else onPageChange?.(0);
    }

    function handlePageChange(next: number) {
        if (onPageChange) onPageChange(next);
        else setInternalPage(next);
    }

    function toggleExpanded(id: string) {
        if (!onExpandedRowChange) return;
        onExpandedRowChange(expandedRowId === id ? null : id);
    }

    const toolbar =
        showDensityToggle ||
        showColumnVisibility ||
        toolbarStart ||
        toolbarEnd ? (
            <Stack
                direction={{ xs: "column", sm: "row" }}
                spacing={1}
                alignItems={{ sm: "center" }}
                justifyContent="space-between"
            >
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                    {toolbarStart}
                </Stack>
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                    {toolbarEnd}
                    {showDensityToggle ? (
                        <ToggleButtonGroup
                            size="small"
                            exclusive
                            value={density}
                            onChange={(_, next: DataTableDensity | null) => {
                                if (next) setDensity(next);
                            }}
                            aria-label="Table density"
                        >
                            <ToggleButton value="compact">Compact</ToggleButton>
                            <ToggleButton value="comfortable">Comfortable</ToggleButton>
                        </ToggleButtonGroup>
                    ) : null}
                    {showColumnVisibility ? (
                        <>
                            <IconButton
                                size="small"
                                aria-label="Column visibility"
                                onClick={(event) => setColumnMenuAnchor(event.currentTarget)}
                            >
                                <ColumnsIcon fontSize="small" />
                            </IconButton>
                            <Menu
                                anchorEl={columnMenuAnchor}
                                open={Boolean(columnMenuAnchor)}
                                onClose={() => setColumnMenuAnchor(null)}
                            >
                                {resolvedColumns
                                    .filter((column) => column.hideable !== false)
                                    .map((column) => {
                                        const hidden = hiddenIds.has(column.id);
                                        return (
                                            <MenuItem
                                                key={column.id}
                                                onClick={() => {
                                                    setHiddenIds((current) => {
                                                        const next = new Set(current);
                                                        if (hidden) next.delete(column.id);
                                                        else next.add(column.id);
                                                        return next;
                                                    });
                                                }}
                                            >
                                                <Checkbox size="small" checked={!hidden} />
                                                {column.label}
                                            </MenuItem>
                                        );
                                    })}
                            </Menu>
                        </>
                    ) : null}
                </Stack>
            </Stack>
        ) : null;

    if (loading) {
        return (
            <Stack spacing={1.5}>
                {toolbar}
                <Skeleton variant="rounded" height={density === "compact" ? 180 : 240} />
            </Stack>
        );
    }

    if (!sortedRows.length) {
        return (
            <Stack spacing={1.5}>
                {toolbar}
                {emptyIcon ? (
                    <EmptyState
                        icon={emptyIcon}
                        title={emptyTitle}
                        description={emptyDescription}
                        action={emptyAction}
                    />
                ) : (
                    <Typography color="text.secondary">{emptyDescription}</Typography>
                )}
            </Stack>
        );
    }

    return (
        <Stack spacing={1.5}>
            {toolbar}
            <TableContainer
                ref={containerRef}
                tabIndex={onKeyDown ? 0 : undefined}
                role="grid"
                aria-label={ariaLabel}
                onKeyDown={onKeyDown}
                sx={{
                    maxHeight,
                    outline: "none",
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 1,
                    width: "100%",
                    maxWidth: "100%",
                    minWidth: 0,
                    overflowX: "auto",
                    WebkitOverflowScrolling: "touch",
                    scrollbarWidth: "thin",
                    "&:focus-visible": {
                        boxShadow: (t) => `0 0 0 2px ${t.palette.primary.main}`,
                    },
                }}
            >
                <Table
                    stickyHeader={stickyHeader}
                    size={tableSizeForDensity(density)}
                    aria-rowcount={visibleRows.length}
                >
                    <TableHead>
                        <TableRow>
                            {renderExpandedRow ? <TableCell padding="checkbox" /> : null}
                            {showSelection ? (
                                <TableCell padding="checkbox">
                                    <Checkbox
                                        size="small"
                                        checked={
                                            visibleRows.length > 0 &&
                                            visibleRows.every((row, index) =>
                                                selectedIds!.includes(getRowId(row, index))
                                            )
                                        }
                                        indeterminate={
                                            visibleRows.some((row, index) =>
                                                selectedIds!.includes(getRowId(row, index))
                                            ) &&
                                            !visibleRows.every((row, index) =>
                                                selectedIds!.includes(getRowId(row, index))
                                            )
                                        }
                                        onChange={onToggleAllOnPage}
                                        inputProps={{ "aria-label": "Select all on page" }}
                                    />
                                </TableCell>
                            ) : null}
                            {visibleColumns.map((column) => (
                                <TableCell
                                    key={column.id}
                                    align={column.align}
                                    padding={column.padding}
                                    sortDirection={
                                        sort?.id === column.id ? sort.direction : false
                                    }
                                    sx={{
                                        width: column.width,
                                        minWidth: column.minWidth,
                                        py: densityTokens.headerPy,
                                        fontWeight: 600,
                                        whiteSpace: "nowrap",
                                        ...stickyCellSx(column.sticky, true),
                                        ...(stickyHeader
                                            ? { zIndex: dataTableDefaults.stickyZIndex.header }
                                            : null),
                                    }}
                                >
                                    {column.sortable ? (
                                        <TableSortLabel
                                            active={sort?.id === column.id}
                                            direction={
                                                sort?.id === column.id ? sort.direction : "asc"
                                            }
                                            onClick={() => handleSort(column.id)}
                                        >
                                            {column.label}
                                        </TableSortLabel>
                                    ) : (
                                        column.label
                                    )}
                                </TableCell>
                            ))}
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {visibleRows.map((row, index) => {
                            const id = getRowId(row, index);
                            const expanded = expandedRowId === id;
                            return (
                                <Fragment key={id}>
                                    <TableRow
                                        hover
                                        selected={selectedRowId === id}
                                        sx={{
                                            cursor: onRowClick ? "pointer" : undefined,
                                            "& > td": { py: densityTokens.cellPy },
                                            minHeight: densityTokens.rowMinHeight,
                                        }}
                                        onClick={() => onRowClick?.(row)}
                                        aria-selected={selectedRowId === id}
                                        tabIndex={-1}
                                    >
                                        {renderExpandedRow ? (
                                            <TableCell
                                                padding="checkbox"
                                                onClick={(event) => event.stopPropagation()}
                                            >
                                                <IconButton
                                                    size="small"
                                                    aria-label={
                                                        expanded ? "Collapse row" : "Expand row"
                                                    }
                                                    onClick={() => toggleExpanded(id)}
                                                >
                                                    {expanded ? (
                                                        <ExpandIcon fontSize="small" />
                                                    ) : (
                                                        <CollapseIcon fontSize="small" />
                                                    )}
                                                </IconButton>
                                            </TableCell>
                                        ) : null}
                                        {showSelection ? (
                                            <TableCell
                                                padding="checkbox"
                                                onClick={(event) => event.stopPropagation()}
                                            >
                                                <Checkbox
                                                    size="small"
                                                    checked={selectedIds!.includes(id)}
                                                    onChange={() => onToggleSelected?.(id)}
                                                    inputProps={{
                                                        "aria-label": `Select row ${id}`,
                                                    }}
                                                />
                                            </TableCell>
                                        ) : null}
                                        {visibleColumns.map((column) => {
                                            const content = column.render(row);
                                            const truncate =
                                                column.truncate === true
                                                    ? dataTableDefaults.truncateChars
                                                    : typeof column.truncate === "number"
                                                      ? column.truncate
                                                      : null;
                                            return (
                                                <TableCell
                                                    key={column.id}
                                                    align={column.align}
                                                    padding={column.padding}
                                                    sx={{
                                                        width: column.width,
                                                        minWidth: column.minWidth,
                                                        maxWidth:
                                                            truncate != null ? 280 : undefined,
                                                        ...stickyCellSx(column.sticky, false),
                                                    }}
                                                >
                                                    {truncate != null &&
                                                    (typeof content === "string" ||
                                                        typeof content === "number" ||
                                                        content == null) ? (
                                                        <TruncatedCell maxChars={truncate}>
                                                            {content}
                                                        </TruncatedCell>
                                                    ) : (
                                                        content
                                                    )}
                                                </TableCell>
                                            );
                                        })}
                                    </TableRow>
                                    {renderExpandedRow ? (
                                        <TableRow>
                                            <TableCell
                                                colSpan={colSpan}
                                                sx={{
                                                    py: 0,
                                                    borderBottom: expanded ? undefined : 0,
                                                }}
                                            >
                                                <Collapse in={expanded} timeout="auto" unmountOnExit>
                                                    <Box sx={{ py: 1.5, px: 1 }}>
                                                        {renderExpandedRow(row)}
                                                    </Box>
                                                </Collapse>
                                            </TableCell>
                                        </TableRow>
                                    ) : null}
                                </Fragment>
                            );
                        })}
                    </TableBody>
                </Table>
            </TableContainer>
            <Stack
                direction={{ xs: "column", sm: "row" }}
                justifyContent="space-between"
                alignItems={{ sm: "center" }}
                spacing={1}
            >
                <Box>{footerStart}</Box>
                <TablePagination
                    component="div"
                    count={count}
                    page={currentPage}
                    onPageChange={(_, next) => handlePageChange(next)}
                    rowsPerPage={pageSize}
                    rowsPerPageOptions={
                        rowsPerPageOptions ??
                        (onPageSizeChange ? [10, 25, 50, 100] : [pageSize])
                    }
                    onRowsPerPageChange={
                        onPageSizeChange
                            ? (event) => {
                                  const next = Number(event.target.value);
                                  setInternalPageSize(next);
                                  onPageSizeChange(next);
                                  handlePageChange(0);
                              }
                            : undefined
                    }
                />
            </Stack>
        </Stack>
    );
}
