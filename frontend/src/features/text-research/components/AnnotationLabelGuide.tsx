import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Box,
    Chip,
    Stack,
    Typography,
} from "@mui/material";
import { ExpandMore as ExpandIcon } from "@mui/icons-material";
import type { AnnotationLabel } from "../types";

/** Compact expandable codebook guidance for the annotation sidebar. */
export function AnnotationLabelGuide({
    labels,
    codebookName,
    codebookVersion,
    frozen,
}: {
    labels: AnnotationLabel[];
    codebookName?: string;
    codebookVersion?: string;
    frozen?: boolean;
}) {
    return (
        <Stack spacing={1}>
            <Typography variant="subtitle2">
                Codebook {codebookName ?? ""}
                {codebookVersion ? ` · v${codebookVersion}` : ""}
                {frozen ? " · frozen" : ""}
            </Typography>
            {labels.map((label) => (
                <Accordion
                    key={label.id}
                    disableGutters
                    elevation={0}
                    sx={{
                        border: 1,
                        borderColor: "divider",
                        borderRadius: 1,
                        "&:before": { display: "none" },
                    }}
                >
                    <AccordionSummary expandIcon={<ExpandIcon />} sx={{ minHeight: 40, px: 1.25 }}>
                        <Stack direction="row" spacing={1} alignItems="center">
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {label.name}
                            </Typography>
                            {label.is_placeholder ? <Chip size="small" label="demo" /> : null}
                        </Stack>
                    </AccordionSummary>
                    <AccordionDetails sx={{ px: 1.25, pt: 0, pb: 1.25 }}>
                        <Stack spacing={0.75}>
                            {label.description ? (
                                <Typography variant="body2" color="text.secondary">
                                    {label.description}
                                </Typography>
                            ) : (
                                <Typography variant="caption" color="text.secondary">
                                    No definition yet.
                                </Typography>
                            )}
                            {label.inclusion_criteria ? (
                                <Typography variant="caption" display="block">
                                    Include: {label.inclusion_criteria}
                                </Typography>
                            ) : null}
                            {label.exclusion_criteria ? (
                                <Typography variant="caption" display="block">
                                    Exclude: {label.exclusion_criteria}
                                </Typography>
                            ) : null}
                            {label.positive_examples?.length ? (
                                <Typography variant="caption" display="block">
                                    + {label.positive_examples.join(" · ")}
                                </Typography>
                            ) : null}
                            {label.negative_examples?.length ? (
                                <Typography variant="caption" display="block">
                                    − {label.negative_examples.join(" · ")}
                                </Typography>
                            ) : null}
                        </Stack>
                    </AccordionDetails>
                </Accordion>
            ))}
            {!labels.length ? (
                <Box>
                    <Typography variant="body2" color="text.secondary">
                        No labels in this codebook.
                    </Typography>
                </Box>
            ) : null}
        </Stack>
    );
}
