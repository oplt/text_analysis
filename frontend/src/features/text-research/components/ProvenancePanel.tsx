import { Box, Stack, Typography } from "@mui/material";
import { AdvancedSettings } from "../../../components/ui/AdvancedSettings";
import { JsonBlock } from "../../../components/ui/JsonBlock";
import { KeyValueList } from "../../../components/ui/KeyValueList";
import type { RunProvenance } from "../../../api/textResearch";
import {
    buildProvenanceSections,
    provenanceRawPayload,
    type ProvenanceRunLike,
} from "../provenanceModel";

type ProvenancePanelProps = {
    run?: ProvenanceRunLike | null;
    detail?: RunProvenance | null;
    compact?: boolean;
    /** Extra actions under reproducibility (e.g. export). */
    actions?: React.ReactNode;
    showRaw?: boolean;
};

/**
 * Readable scientific provenance — visually secondary to charts/metrics.
 */
export function ProvenancePanel({
    run,
    detail,
    compact = false,
    actions,
    showRaw = !compact,
}: ProvenancePanelProps) {
    const sections = buildProvenanceSections({ run, detail });

    if (!sections.length && !showRaw) {
        return (
            <Typography variant="body2" color="text.secondary">
                Provenance details are not available for this run yet.
            </Typography>
        );
    }

    return (
        <Stack spacing={compact ? 1 : 1.5}>
            {sections.map((section) => (
                <Box key={section.id}>
                    <Typography variant="subtitle2" gutterBottom>
                        {section.title}
                    </Typography>
                    <KeyValueList
                        dense
                        showHelp
                        items={section.fields.map((item) => ({
                            key: item.key,
                            label: item.label,
                            value: item.value,
                            helpTermId: item.helpTermId,
                        }))}
                    />
                </Box>
            ))}
            {actions}
            {showRaw ? (
                <AdvancedSettings title="Raw provenance details">
                    <JsonBlock data={provenanceRawPayload({ run, detail })} />
                </AdvancedSettings>
            ) : null}
        </Stack>
    );
}
