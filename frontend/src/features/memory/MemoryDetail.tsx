import {
    Alert,
    Box,
    Button,
    Chip,
    Stack,
    Typography,
} from "@mui/material";
import { DeleteOutline as DeleteIcon } from "@mui/icons-material";
import { QueryBoundary } from "../../components/ui/QueryBoundary";
import { SectionCard } from "../../components/ui/SectionCard";
import type { MemoryItem } from "../../api/memory";

type MemoryDetailProps = {
    item: MemoryItem | null;
    isLoading: boolean;
    isError: boolean;
    error: unknown;
    onRetry: () => void;
    onForget: () => void;
};

export function MemoryDetail({
    item,
    isLoading,
    isError,
    error,
    onRetry,
    onForget,
}: MemoryDetailProps) {
    return (
        <SectionCard
            title="Memory detail"
            description="Source, confidence, privacy, and timestamps."
            action={
                <Button
                    size="small"
                    color="error"
                    startIcon={<DeleteIcon />}
                    disabled={!item}
                    onClick={onForget}
                >
                    Forget
                </Button>
            }
        >
            <QueryBoundary
                isLoading={isLoading}
                isError={isError}
                error={error}
                onRetry={onRetry}
            >
                {!item ? (
                    <Typography variant="body2" color="text.secondary">
                        Select a memory to inspect.
                    </Typography>
                ) : (
                    <Stack spacing={1.5}>
                        <Typography variant="body1">{item.content}</Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                            <Chip size="small" label={item.metadata.memory_level} />
                            <Chip size="small" variant="outlined" label={item.metadata.memory_type} />
                            <Chip size="small" variant="outlined" label={item.metadata.privacy} />
                            <Chip
                                size="small"
                                variant="outlined"
                                label={`confidence ${item.metadata.confidence.toFixed(2)}`}
                            />
                        </Stack>
                        <Box>
                            <Typography variant="caption" color="text.secondary" display="block">
                                Source: {item.metadata.source}
                                {item.metadata.source_ref
                                    ? ` · ${item.metadata.source_ref}`
                                    : ""}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block">
                                Agent: {item.metadata.agent_id}
                                {item.metadata.project_id
                                    ? ` · project ${item.metadata.project_id}`
                                    : ""}
                                {item.metadata.run_id ? ` · run ${item.metadata.run_id}` : ""}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block">
                                Created:{" "}
                                {item.metadata.created_at
                                    ? new Date(item.metadata.created_at).toLocaleString()
                                    : "—"}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block">
                                Last seen:{" "}
                                {item.metadata.last_seen_at
                                    ? new Date(item.metadata.last_seen_at).toLocaleString()
                                    : "—"}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block">
                                Last confirmed:{" "}
                                {item.metadata.last_confirmed_at
                                    ? new Date(item.metadata.last_confirmed_at).toLocaleString()
                                    : "—"}
                            </Typography>
                        </Box>
                        {item.score != null ? (
                            <Alert severity="info">Search score: {item.score.toFixed(3)}</Alert>
                        ) : null}
                    </Stack>
                )}
            </QueryBoundary>
        </SectionCard>
    );
}
