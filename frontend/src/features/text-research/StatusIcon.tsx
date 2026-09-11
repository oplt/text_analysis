import {
    CheckCircle as CompleteIcon,
    Circle as CurrentIcon,
    ErrorOutline as BlockedIcon,
    RadioButtonUnchecked as IncompleteIcon,
    WarningAmber as WarningIcon,
} from "@mui/icons-material";
import type { WorkflowStatus } from "./workflow";

export function StatusIcon({
    status,
    size = "small",
}: {
    status: WorkflowStatus;
    size?: "inherit" | "small" | "medium" | "large";
}) {
    switch (status) {
        case "complete":
            return <CompleteIcon color="success" fontSize={size} />;
        case "current":
            return <CurrentIcon color="primary" fontSize={size} />;
        case "warning":
            return <WarningIcon color="warning" fontSize={size} />;
        case "blocked":
            return <BlockedIcon color="disabled" fontSize={size} />;
        default:
            return <IncompleteIcon color="action" fontSize={size} />;
    }
}
