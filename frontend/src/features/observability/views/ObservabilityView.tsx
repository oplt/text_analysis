import { Alert, Button, Stack } from "@mui/material";
import { OpenInNew as OpenInNewIcon } from "@mui/icons-material";
import { SettingsTabs } from "../../../components/layout/SettingsTabs";
import { PageShell } from "../../../components/ui/PageShell";
import { InvestigationContextFields } from "../components/InvestigationContextFields";
import { ObservabilityHealthGrid } from "../components/ObservabilityHealthGrid";
import { ObservabilityShortcuts } from "../components/ObservabilityShortcuts";
import { useObservabilityView } from "../hooks/useObservabilityView";
import { buildGrafanaUrl, openExternalUrl } from "../urlBuilders";

export default function ObservabilityPage() {
    const model = useObservabilityView();
    const links = model.linksQuery.data;

    return (
        <PageShell maxWidth="xl">
            <SettingsTabs />
            <Stack direction="row" justifyContent="flex-end" sx={{ mb: 2 }}>
                <Button
                    variant="contained"
                    startIcon={<OpenInNewIcon />}
                    disabled={!model.technicalAccess || !links?.dashboards.application_overview.url}
                    onClick={() => {
                        const url = buildGrafanaUrl(links?.dashboards.application_overview.url, model.context);
                        if (url) openExternalUrl(url);
                    }}
                >
                    Open Grafana
                </Button>
            </Stack>
            {model.linksQuery.isError && (
                <Alert severity="warning">
                    Observability links are unavailable. Health checks can still render when the status endpoint responds.
                </Alert>
            )}
            <InvestigationContextFields filters={model.filters} setters={model.setters} />
            <ObservabilityHealthGrid healthItems={model.healthItems} statusQuery={model.statusQuery} />
            <ObservabilityShortcuts context={model.context} isAdmin={model.isAdmin} linksQuery={model.linksQuery} />
        </PageShell>
    );
}
