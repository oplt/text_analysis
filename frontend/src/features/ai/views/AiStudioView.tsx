import { Box, Button, Skeleton, Stack } from "@mui/material";
import { Approval as ReviewIcon, Dataset as DatasetIcon, Description as DocumentIcon, PsychologyAlt as PromptIcon } from "@mui/icons-material";
import { SettingsTabs } from "../../../components/layout/SettingsTabs";
import { PageShell } from "../../../components/ui/PageShell";
import { QueryErrorAlert } from "../../../components/ui/QueryBoundary";
import { StatCard } from "../../../components/ui/StatCard";
import { StatCardSkeletonGrid } from "../../../components/ui/StatCardSkeletonGrid";
import { PromptLibraryPanel } from "../components/PromptLibraryPanel";
import { RetrievalDocumentsPanel } from "../components/RetrievalDocumentsPanel";
import { ReviewsEvaluationsPanel } from "../components/ReviewsEvaluationsPanel";
import { RunPlaygroundPanel } from "../components/RunPlaygroundPanel";
import { VersionBuilderPanel } from "../components/VersionBuilderPanel";
import { useAiStudioView } from "../hooks/useAiStudioView";

export default function AiStudioPage() {
    const m = useAiStudioView();
    if (m.isLoading) return <PageShell maxWidth="xl"><SettingsTabs /><StatCardSkeletonGrid /><Stack spacing={2} sx={{ mt: 2 }}><Skeleton variant="rounded" height={320} /><Skeleton variant="rounded" height={280} /></Stack></PageShell>;
    if (m.isError || !m.overview) return <PageShell maxWidth="xl"><SettingsTabs /><QueryErrorAlert error={m.error ?? new Error("Failed to load AI studio overview.")} fallback="Failed to load AI studio overview." onRetry={() => void m.refetch()} /></PageShell>;
    const sections = [
        ["ai-prompts", "Prompts"],
        ["ai-runs", "Run"],
        ["ai-versions", "Versions"],
        ["ai-documents", "Documents"],
        ["ai-evaluations", "Evaluations"],
    ] as const;
    return <PageShell maxWidth="xl"><SettingsTabs />
        <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", sm: "repeat(2,1fr)", xl: "repeat(4,1fr)" } }}>
            <StatCard label="Prompt templates" value={m.promptTemplates.length} description="Reusable prompts with version history" icon={<PromptIcon />} />
            <StatCard label="Documents" value={m.documents.length} description="Indexed retrieval sources" icon={<DocumentIcon />} color="secondary" />
            <StatCard label="Pending reviews" value={m.reviews.filter((item) => item.status === "pending").length} description="Runs waiting for human review" icon={<ReviewIcon />} color="warning" />
            <StatCard label="Datasets" value={m.datasets.length} description="Saved evaluation datasets" icon={<DatasetIcon />} color="success" />
        </Box>
        <Stack
            component="nav"
            aria-label="AI Studio sections"
            direction="row"
            spacing={1}
            sx={{ mt: 2, overflowX: "auto", pb: 0.5, display: { xl: "none" } }}
        >
            {sections.map(([id, label]) => <Button key={id} component="a" href={`#${id}`} size="small" variant="outlined" sx={{ flexShrink: 0 }}>{label}</Button>)}
        </Stack>
        <Box sx={{ mt: 2, display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", xl: "1.1fr 0.9fr" }, alignItems: "start" }}>
            <Stack spacing={2}><Box id="ai-prompts" sx={{ scrollMarginTop: 88 }}><PromptLibraryPanel m={m} /></Box><Box id="ai-runs" sx={{ scrollMarginTop: 88 }}><RunPlaygroundPanel m={m} /></Box></Stack>
            <Stack spacing={2}><Box id="ai-versions" sx={{ scrollMarginTop: 88 }}><VersionBuilderPanel m={m} /></Box><Box id="ai-documents" sx={{ scrollMarginTop: 88 }}><RetrievalDocumentsPanel m={m} /></Box><Box id="ai-evaluations" sx={{ scrollMarginTop: 88 }}><ReviewsEvaluationsPanel m={m} /></Box></Stack>
        </Box>
    </PageShell>;
}
