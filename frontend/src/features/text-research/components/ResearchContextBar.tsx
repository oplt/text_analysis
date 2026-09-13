import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import {
    Box,
    Button,
    Chip,
    Collapse,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    FormControl,
    IconButton,
    InputLabel,
    MenuItem,
    Select,
    Stack,
    Tooltip,
    Typography,
    useMediaQuery,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import {
    ExpandLess as CollapseIcon,
    ExpandMore as ExpandIcon,
    Link as ProjectIcon,
} from "@mui/icons-material";
import { Link as RouterLink } from "react-router-dom";
import { useSnackbar } from "../../../app/snackbarContext";
import { FormGrid } from "../../../components/ui/FormGrid";
import { useResearchContext } from "../hooks/useResearchContext";
import { UNIT_TYPE_OPTIONS, type UnitType } from "../types";

type PendingChange =
    | { kind: "corpus"; nextId: string; nextLabel: string }
    | { kind: "codebook"; nextId: string; nextLabel: string }
    | { kind: "unit"; next: UnitType; nextLabel: string };

export type ResearchContextBarVariant = "full" | "summary";

type ResearchContextBarProps = {
    /** Project display name (optional; falls back to project id). */
    projectName?: string;
    /**
     * `full` — editable compact selectors (layout).
     * `summary` — read-only chips (e.g. Ask Corpus context tab).
     */
    variant?: ResearchContextBarVariant;
    /** Primary workspace actions (e.g. Next stage, Ask Corpus) grouped with context. */
    actions?: ReactNode;
};

function FieldLabel({ children }: { children: React.ReactNode }) {
    return (
        <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: "block", mb: 0.35, lineHeight: 1.2 }}
        >
            {children}
        </Typography>
    );
}

/**
 * Compact research scope bar: project, corpus, unit type, codebook.
 * Lives below app chrome and above research navigation (Phase 4).
 */
export function ResearchContextBar({
    projectName,
    variant = "full",
    actions,
}: ResearchContextBarProps) {
    const theme = useTheme();
    const isNarrow = useMediaQuery(theme.breakpoints.down("md"));
    const ctx = useResearchContext();
    const { showToast } = useSnackbar();
    const [expanded, setExpanded] = useState(!isNarrow);
    const [pending, setPending] = useState<PendingChange | null>(null);
    const [flash, setFlash] = useState(false);
    const flashTimer = useRef<number | null>(null);
    const formId = useId();

    useEffect(() => {
        setExpanded(!isNarrow);
    }, [isNarrow]);

    useEffect(() => {
        return () => {
            if (flashTimer.current != null) window.clearTimeout(flashTimer.current);
        };
    }, []);

    function pulseChanged(message: string) {
        setFlash(true);
        if (flashTimer.current != null) window.clearTimeout(flashTimer.current);
        flashTimer.current = window.setTimeout(() => setFlash(false), 1200);
        showToast({ message, severity: "info" });
    }

    function requestCorpusChange(nextId: string) {
        if (!nextId || nextId === ctx.selectedCorpusId) return;
        const next = ctx.corpora.find((c) => c.id === nextId);
        const nextLabel = next?.name ?? nextId;
        if (ctx.selectedCorpusId) {
            setPending({ kind: "corpus", nextId, nextLabel });
            return;
        }
        ctx.setSelectedCorpusId(nextId);
        pulseChanged(`Corpus set to “${nextLabel}”.`);
    }

    function requestCodebookChange(nextId: string) {
        if (nextId === ctx.selectedCodebookId) return;
        const next = ctx.codebooks.find((c) => c.id === nextId);
        const nextLabel = next
            ? `${next.name} (v${next.version})`
            : nextId || "None";
        if (ctx.selectedCodebookId && nextId) {
            setPending({ kind: "codebook", nextId, nextLabel });
            return;
        }
        ctx.setSelectedCodebookId(nextId);
        pulseChanged(nextId ? `Codebook set to “${nextLabel}”.` : "Codebook cleared.");
    }

    function requestUnitChange(next: UnitType) {
        if (next === ctx.unitType) return;
        const nextLabel = UNIT_TYPE_OPTIONS.find((o) => o.value === next)?.label ?? next;
        setPending({ kind: "unit", next, nextLabel });
    }

    function confirmPending() {
        if (!pending) return;
        if (pending.kind === "corpus") {
            ctx.setSelectedCorpusId(pending.nextId);
            pulseChanged(`Corpus switched to “${pending.nextLabel}”.`);
        } else if (pending.kind === "codebook") {
            ctx.setSelectedCodebookId(pending.nextId);
            pulseChanged(`Codebook switched to “${pending.nextLabel}”.`);
        } else {
            ctx.setUnitType(pending.next);
            pulseChanged(`Unit type switched to ${pending.nextLabel}.`);
        }
        setPending(null);
    }

    const displayProject = projectName || ctx.projectId;
    const corpusLabel = ctx.selectedCorpus?.name ?? "Not selected";
    const codebookLabel = ctx.selectedCodebook
        ? `${ctx.selectedCodebook.name} v${ctx.selectedCodebook.version}`
        : "Not selected";
    const unitLabel =
        UNIT_TYPE_OPTIONS.find((o) => o.value === ctx.unitType)?.label ?? ctx.unitType;

    const summaryChips = (
        <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap alignItems="center">
            <Chip
                size="small"
                icon={<ProjectIcon />}
                label={displayProject}
                component={RouterLink}
                to={`/projects/${ctx.projectId}`}
                clickable
                variant="outlined"
                sx={{ maxWidth: 220 }}
            />
            <Chip size="small" label={`Corpus: ${corpusLabel}`} variant="outlined" />
            <Chip size="small" label={`Unit: ${unitLabel}`} variant="outlined" />
            <Chip size="small" label={`Codebook: ${codebookLabel}`} variant="outlined" />
        </Stack>
    );

    if (variant === "summary") {
        return (
            <Box
                aria-label="Research context summary"
                sx={{
                    px: 1,
                    py: 0.75,
                    borderRadius: 1.5,
                    border: 1,
                    borderColor: "divider",
                    bgcolor: "action.hover",
                }}
            >
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
                    Active scope (change in the context bar above)
                </Typography>
                {summaryChips}
            </Box>
        );
    }

    const editors = (
        <FormGrid columns="4" gap={1.25}>
            <Box sx={{ minWidth: 0 }}>
                <FieldLabel>Project</FieldLabel>
                <Button
                    fullWidth
                    size="small"
                    variant="outlined"
                    component={RouterLink}
                    to={`/projects/${ctx.projectId}`}
                    startIcon={<ProjectIcon fontSize="small" />}
                    sx={{
                        justifyContent: "flex-start",
                        textTransform: "none",
                        fontWeight: 600,
                        maxWidth: "100%",
                    }}
                >
                    <Typography noWrap component="span" variant="body2">
                        {displayProject}
                    </Typography>
                </Button>
            </Box>

            <FormControl
                size="small"
                fullWidth
                disabled={ctx.corporaLoading || ctx.corpora.length === 0}
            >
                <InputLabel id={`${formId}-corpus`}>Corpus</InputLabel>
                <Select
                    labelId={`${formId}-corpus`}
                    label="Corpus"
                    value={ctx.selectedCorpusId || ""}
                    onChange={(e) => requestCorpusChange(String(e.target.value))}
                >
                    {ctx.corpora.length === 0 ? (
                        <MenuItem value="" disabled>
                            No corpora
                        </MenuItem>
                    ) : null}
                    {ctx.corpora.map((corpus) => (
                        <MenuItem key={corpus.id} value={corpus.id}>
                            {corpus.name}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>

            <FormControl size="small" fullWidth>
                <InputLabel id={`${formId}-unit`}>Unit type</InputLabel>
                <Select
                    labelId={`${formId}-unit`}
                    label="Unit type"
                    value={ctx.unitType}
                    onChange={(e) => requestUnitChange(e.target.value as UnitType)}
                >
                    {UNIT_TYPE_OPTIONS.map((option) => (
                        <MenuItem key={option.value} value={option.value}>
                            {option.label}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>

            <FormControl size="small" fullWidth disabled={ctx.codebooksLoading}>
                <InputLabel id={`${formId}-codebook`}>Codebook</InputLabel>
                <Select
                    labelId={`${formId}-codebook`}
                    label="Codebook"
                    value={ctx.selectedCodebookId || ""}
                    onChange={(e) => requestCodebookChange(String(e.target.value))}
                    displayEmpty
                >
                    <MenuItem value="">
                        <em>None</em>
                    </MenuItem>
                    {ctx.codebooks.map((codebook) => (
                        <MenuItem key={codebook.id} value={codebook.id}>
                            {codebook.name} (v{codebook.version}
                            {codebook.is_frozen ? " · frozen" : ""})
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>

            {ctx.codebooks.length === 0 ? (
                <Button
                    size="small"
                    variant="text"
                    component={RouterLink}
                    to={`/research/${ctx.projectId}/codebook`}
                >
                    Create codebook
                </Button>
            ) : null}
        </FormGrid>
    );

    return (
        <>
            <Box
                component="section"
                aria-label="Research context"
                sx={{
                    px: { xs: 1.25, md: 1.5 },
                    py: { xs: 1, md: 1.1 },
                    borderRadius: 2,
                    border: 1,
                    borderColor: flash ? "primary.main" : "divider",
                    bgcolor: flash ? "action.selected" : "background.paper",
                    transition: (t) =>
                        t.transitions.create(["border-color", "background-color"], {
                            duration: 280,
                        }),
                }}
            >
                <Stack
                    direction="row"
                    alignItems="center"
                    justifyContent="space-between"
                    spacing={1}
                    flexWrap="wrap"
                    useFlexGap
                    sx={{ mb: isNarrow && !expanded ? 0 : 1 }}
                >
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                        Research context
                    </Typography>
                    <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                        {actions}
                        {isNarrow ? (
                            <Tooltip title={expanded ? "Collapse context" : "Edit context"}>
                                <IconButton
                                    size="small"
                                    aria-label={
                                        expanded
                                            ? "Collapse research context"
                                            : "Expand research context"
                                    }
                                    aria-expanded={expanded}
                                    onClick={() => setExpanded((v) => !v)}
                                >
                                    {expanded ? (
                                        <CollapseIcon fontSize="small" />
                                    ) : (
                                        <ExpandIcon fontSize="small" />
                                    )}
                                </IconButton>
                            </Tooltip>
                        ) : null}
                    </Stack>
                </Stack>

                {isNarrow && !expanded ? summaryChips : null}

                <Collapse in={!isNarrow || expanded} unmountOnExit={false}>
                    {editors}
                </Collapse>
            </Box>

            <Dialog
                open={Boolean(pending)}
                onClose={() => setPending(null)}
                fullWidth
                maxWidth="xs"
                aria-labelledby={`${formId}-confirm-title`}
            >
                <DialogTitle id={`${formId}-confirm-title`}>Change research context?</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        {pending?.kind === "corpus"
                            ? `Switch corpus to “${pending.nextLabel}”? Analysis results and panels will refresh for the new corpus.`
                            : pending?.kind === "codebook"
                              ? `Switch codebook to “${pending.nextLabel}”? Annotation and reliability views use this codebook.`
                              : pending
                                ? `Switch unit type to ${pending.nextLabel}? Segmentation-dependent tools will use this unit size.`
                                : null}
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setPending(null)}>Cancel</Button>
                    <Button variant="contained" onClick={confirmPending} autoFocus>
                        Change
                    </Button>
                </DialogActions>
            </Dialog>
        </>
    );
}
