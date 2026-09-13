import { Box, type SxProps, type Theme } from "@mui/material";
import { layoutSpacing } from "./layoutTokens";

/**
 * Form field row layouts (12-column semantic spans on md+).
 * Mobile always stacks to a single column.
 */
export type FormGridColumns = "1" | "2" | "3" | "4" | "6-6" | "4-4-4" | "8-4" | "4-8" | "3-9" | "9-3";

const FORM_GRID_COLUMNS: Record<
    FormGridColumns,
    { xs: string; sm?: string; md?: string; lg?: string }
> = {
    "1": { xs: "minmax(0, 1fr)" },
    "2": {
        xs: "minmax(0, 1fr)",
        sm: "repeat(2, minmax(0, 1fr))",
    },
    "3": {
        xs: "minmax(0, 1fr)",
        sm: "repeat(2, minmax(0, 1fr))",
        md: "repeat(2, minmax(0, 1fr))",
        lg: "repeat(3, minmax(0, 1fr))",
    },
    "4": {
        xs: "minmax(0, 1fr)",
        sm: "repeat(2, minmax(0, 1fr))",
        md: "repeat(2, minmax(0, 1fr))",
        lg: "repeat(4, minmax(0, 1fr))",
    },
    "6-6": {
        xs: "minmax(0, 1fr)",
        sm: "repeat(2, minmax(0, 1fr))",
    },
    "4-4-4": {
        xs: "minmax(0, 1fr)",
        sm: "repeat(2, minmax(0, 1fr))",
        md: "repeat(2, minmax(0, 1fr))",
        lg: "repeat(3, minmax(0, 1fr))",
    },
    "8-4": {
        xs: "minmax(0, 1fr)",
        md: "minmax(0, 2fr) minmax(0, 1fr)",
    },
    "4-8": {
        xs: "minmax(0, 1fr)",
        md: "minmax(0, 1fr) minmax(0, 2fr)",
    },
    "3-9": {
        xs: "minmax(0, 1fr)",
        md: "minmax(0, 1fr) minmax(0, 3fr)",
    },
    "9-3": {
        xs: "minmax(0, 1fr)",
        md: "minmax(0, 3fr) minmax(0, 1fr)",
    },
};

type FormGridProps = {
    children: React.ReactNode;
    columns?: FormGridColumns;
    gap?: number;
    sx?: SxProps<Theme>;
};

/**
 * Balanced responsive form/control grid. Prefer over ad-hoc Stacks with large minWidths.
 */
export function FormGrid({
    children,
    columns = "4-4-4",
    gap = layoutSpacing.gridGap,
    sx,
}: FormGridProps) {
    const template = FORM_GRID_COLUMNS[columns];
    return (
        <Box
            sx={[
                {
                    display: "grid",
                    gap,
                    gridTemplateColumns: template,
                    alignItems: "start",
                    width: "100%",
                    "& > .MuiFormControl-root, & > .MuiTextField-root": {
                        width: "100%",
                        minWidth: 0,
                        maxWidth: "100%",
                    },
                },
                ...(Array.isArray(sx) ? sx : sx ? [sx] : []),
            ]}
        >
            {children}
        </Box>
    );
}

/** Shared sx for form controls that should flex without dominating a row. */
export const formControlSx = {
    width: { xs: "100%", sm: "auto" },
    minWidth: { xs: 0, sm: 140 },
    maxWidth: "100%",
    flex: { sm: "1 1 140px" },
} as const;
