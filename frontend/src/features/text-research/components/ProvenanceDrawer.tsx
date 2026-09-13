import { useId, useState } from "react";
import {
    Box,
    Drawer,
    IconButton,
    Stack,
    Tooltip,
    Typography,
} from "@mui/material";
import {
    Close as CloseIcon,
    InfoOutlined as InfoIcon,
} from "@mui/icons-material";
import { useQuery } from "@tanstack/react-query";
import { getRunProvenance, type RunProvenance } from "../../../api/textResearch";
import { QueryBoundary } from "../../../components/ui/QueryBoundary";
import { queryKeys } from "../../../config/queryKeys";
import { QUERY_STALE_TIMES } from "../../../config/queryTiming";
import type { ProvenanceRunLike } from "../provenanceModel";
import { ProvenancePanel } from "./ProvenancePanel";

type ProvenanceDrawerProps = {
    run: ProvenanceRunLike;
    /** Prefetched /provenance payload when already loaded by the parent. */
    detail?: RunProvenance | null;
    /** Fetch /runs/{id}/provenance when the drawer opens (default true). */
    fetchDetail?: boolean;
    title?: string;
    tooltip?: string;
    /** Icon-only trigger (default) or custom trigger element. */
    trigger?: React.ReactNode;
    /** Controlled open state (optional). */
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
    actions?: React.ReactNode;
};

/**
 * Secondary provenance interaction shared across research surfaces.
 */
export function ProvenanceDrawer({
    run,
    detail: detailProp,
    fetchDetail = true,
    title = "Provenance & reproducibility",
    tooltip = "Provenance and reproducibility",
    trigger,
    open: openProp,
    onOpenChange,
    actions,
}: ProvenanceDrawerProps) {
    const titleId = useId();
    const [internalOpen, setInternalOpen] = useState(false);
    const open = openProp ?? internalOpen;
    const setOpen = (next: boolean) => {
        onOpenChange?.(next);
        if (openProp == null) setInternalOpen(next);
    };

    const detailQuery = useQuery({
        queryKey: queryKeys.textResearch.runProvenance(run.id),
        queryFn: ({ signal }) => getRunProvenance(run.id, signal),
        enabled: open && fetchDetail && detailProp == null,
        staleTime: QUERY_STALE_TIMES.researchProvenance,
    });

    const detail = detailProp ?? detailQuery.data ?? null;

    const defaultTrigger = (
        <Tooltip title={tooltip}>
            <IconButton
                aria-label={tooltip}
                aria-haspopup="dialog"
                aria-expanded={open}
                size="small"
                onClick={() => setOpen(true)}
            >
                <InfoIcon fontSize="small" />
            </IconButton>
        </Tooltip>
    );

    return (
        <>
            {trigger ? (
                <Box
                    component="button"
                    type="button"
                    aria-label={tooltip}
                    aria-haspopup="dialog"
                    aria-expanded={open}
                    onClick={() => setOpen(true)}
                    sx={{
                        display: "inline-flex",
                        alignItems: "center",
                        cursor: "pointer",
                        border: 0,
                        background: "transparent",
                        p: 0,
                        m: 0,
                        font: "inherit",
                        color: "inherit",
                        "&:focus-visible": {
                            outline: (theme) => `2px solid ${theme.palette.primary.main}`,
                            outlineOffset: 2,
                            borderRadius: 1,
                        },
                    }}
                >
                    {trigger}
                </Box>
            ) : (
                defaultTrigger
            )}
            <Drawer
                anchor="right"
                open={open}
                onClose={() => setOpen(false)}
                ModalProps={{
                    keepMounted: false,
                }}
                PaperProps={{
                    "aria-labelledby": titleId,
                }}
            >
                <Stack
                    spacing={2}
                    sx={{
                        width: { xs: "min(100vw, 360px)", sm: 420 },
                        p: 3,
                        height: "100%",
                        boxSizing: "border-box",
                    }}
                >
                    <Stack
                        direction="row"
                        spacing={1}
                        alignItems="flex-start"
                        justifyContent="space-between"
                    >
                        <Typography id={titleId} variant="h6" component="h2">
                            {title}
                        </Typography>
                        <IconButton
                            aria-label="Close provenance drawer"
                            size="small"
                            onClick={() => setOpen(false)}
                        >
                            <CloseIcon fontSize="small" />
                        </IconButton>
                    </Stack>
                    <Typography variant="body2" color="text.secondary">
                        Scientific identity for this run — secondary to results, required for
                        replay and exact reproduce.
                    </Typography>
                    <Box sx={{ flex: 1, minHeight: 0, overflow: "auto" }}>
                        {fetchDetail && detailProp == null ? (
                            <QueryBoundary
                                isLoading={detailQuery.isLoading}
                                isError={detailQuery.isError}
                                error={detailQuery.error}
                                onRetry={() => void detailQuery.refetch()}
                                variant="inline"
                            >
                                <ProvenancePanel run={run} detail={detail} actions={actions} />
                            </QueryBoundary>
                        ) : (
                            <ProvenancePanel run={run} detail={detail} actions={actions} />
                        )}
                    </Box>
                </Stack>
            </Drawer>
        </>
    );
}
