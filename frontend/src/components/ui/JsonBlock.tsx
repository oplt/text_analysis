import { Box, Button, Stack } from "@mui/material";
import { ContentCopy as CopyIcon } from "@mui/icons-material";
import { useSnackbar } from "../../app/snackbarContext";
import { copyTextToClipboard, stringifyPretty } from "./jsonDisplay";
import { surfaceCode } from "./themeSurfaces";

type JsonBlockProps = {
    data: unknown;
    showCopy?: boolean;
    maxHeight?: number;
};

/**
 * Expert/debug JSON surface — keep collapsed under Advanced / Raw / Debug.
 */
export function JsonBlock({ data, showCopy = true, maxHeight = 420 }: JsonBlockProps) {
    const { showToast } = useSnackbar();
    const text = stringifyPretty(data);

    return (
        <Stack spacing={1}>
            {showCopy ? (
                <Button
                    size="small"
                    variant="outlined"
                    startIcon={<CopyIcon />}
                    onClick={() => {
                        void copyTextToClipboard(text).then((ok) => {
                            showToast({
                                message: ok ? "JSON copied." : "Could not copy JSON.",
                                severity: ok ? "success" : "warning",
                            });
                        });
                    }}
                    sx={{ alignSelf: "flex-start" }}
                >
                    Copy JSON
                </Button>
            ) : null}
            <Box
                component="pre"
                sx={(theme) => ({
                    m: 0,
                    p: 2,
                    borderRadius: 2,
                    overflow: "auto",
                    maxHeight,
                    fontSize: 12,
                    lineHeight: 1.5,
                    fontFamily:
                        "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                    color: "text.primary",
                    border: 1,
                    borderColor: "divider",
                    bgcolor: surfaceCode(theme),
                })}
            >
                {text}
            </Box>
        </Stack>
    );
}
