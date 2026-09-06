import { useMemo, useState, type PropsWithChildren } from "react";
import { ThemeProvider, CssBaseline, useMediaQuery } from "@mui/material";
import { QueryClientProvider } from "@tanstack/react-query";
import { LocalizationProvider } from "@mui/x-date-pickers";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { darkTheme, lightTheme, type ColorMode } from "./theme";
import { queryClient } from "../config/queryClient";
import { AuthProvider } from "../features/auth/context/AuthContext";
import { SnackbarProvider } from "./SnackbarProvider";
import { ColorModeContext } from "./colorModeContext";

export function AppProviders({ children }: PropsWithChildren) {
    const [colorMode, setColorMode] = useState<ColorMode>("system");
    const prefersDark = useMediaQuery("(prefers-color-scheme: dark)");

    const theme = useMemo(() => {
        if (colorMode === "system") return prefersDark ? darkTheme : lightTheme;
        return colorMode === "dark" ? darkTheme : lightTheme;
    }, [colorMode, prefersDark]);

    return (
        <ColorModeContext.Provider value={{ colorMode, setColorMode }}>
            <QueryClientProvider client={queryClient}>
                <LocalizationProvider dateAdapter={AdapterDayjs}>
                    <ThemeProvider theme={theme}>
                        <CssBaseline />
                        <AuthProvider>
                            <SnackbarProvider>
                                {children}
                            </SnackbarProvider>
                        </AuthProvider>
                    </ThemeProvider>
                </LocalizationProvider>
            </QueryClientProvider>
        </ColorModeContext.Provider>
    );
}
