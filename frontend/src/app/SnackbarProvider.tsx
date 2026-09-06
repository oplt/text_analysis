import { useCallback, useState, type PropsWithChildren } from "react";
import { Alert, Snackbar } from "@mui/material";
import { SnackbarContext, type ToastOptions } from "./snackbarContext";

export function SnackbarProvider({ children }: PropsWithChildren) {
    const [open, setOpen] = useState(false);
    const [opts, setOpts] = useState<ToastOptions>({ message: "", severity: "info" });

    const showToast = useCallback((o: ToastOptions) => {
        setOpts(o);
        setOpen(true);
    }, []);

    return (
        <SnackbarContext.Provider value={{ showToast }}>
            {children}
            <Snackbar
                open={open}
                autoHideDuration={4000}
                onClose={() => setOpen(false)}
                anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
            >
                <Alert
                    severity={opts.severity ?? "info"}
                    onClose={() => setOpen(false)}
                    variant="filled"
                    sx={{ width: "100%" }}
                >
                    {opts.message}
                </Alert>
            </Snackbar>
        </SnackbarContext.Provider>
    );
}
