import { alpha, createTheme, type PaletteMode } from "@mui/material/styles";
import { colors, fonts, motion, radii } from "./designTokens";

const transition = `border-color ${motion.durationMs}ms, background-color ${motion.durationMs}ms, color ${motion.durationMs}ms, box-shadow 250ms`;

function buildTheme(mode: PaletteMode) {
    const isDark = mode === "dark";

    const theme = createTheme({
        palette: {
            mode,
            primary: {
                main: colors.electricBlue,
                light: "#5A7FE8",
                dark: "#3259C4",
                contrastText: colors.white,
            },
            secondary: {
                main: isDark ? colors.graphite : colors.graphite,
                light: colors.pewter,
                dark: colors.carbonDark,
                contrastText: colors.white,
            },
            success: {
                main: isDark ? "#86d8a7" : "#2f8f57",
            },
            warning: {
                main: isDark ? "#e8b87a" : "#c4842d",
            },
            error: {
                main: isDark ? "#f87171" : "#dc2626",
            },
            info: {
                main: colors.electricBlue,
            },
            background: {
                default: isDark ? colors.carbonDark : colors.white,
                paper: isDark ? "#1E2128" : colors.white,
            },
            text: {
                primary: isDark ? colors.white : colors.carbonDark,
                secondary: isDark ? alpha(colors.white, 0.72) : colors.graphite,
                disabled: colors.silverFog,
            },
            divider: isDark ? alpha(colors.white, 0.12) : colors.cloudGray,
        },
        shape: {
            borderRadius: radii.button,
        },
        typography: {
            fontFamily: fonts.text,
            fontWeightLight: 400,
            fontWeightRegular: 400,
            fontWeightMedium: 500,
            fontWeightBold: 500,
            h1: {
                fontFamily: fonts.display,
                fontSize: "2.5rem",
                fontWeight: 500,
                letterSpacing: "normal",
                lineHeight: 1.2,
            },
            h2: {
                fontFamily: fonts.display,
                fontSize: "2rem",
                fontWeight: 500,
                letterSpacing: "normal",
                lineHeight: 1.2,
            },
            h3: {
                fontFamily: fonts.display,
                fontSize: "1.75rem",
                fontWeight: 500,
                letterSpacing: "normal",
                lineHeight: 1.2,
            },
            h4: {
                fontFamily: fonts.display,
                fontSize: "1.375rem",
                fontWeight: 500,
                letterSpacing: "normal",
                lineHeight: 1.18,
            },
            h5: {
                fontSize: "1.0625rem",
                fontWeight: 500,
                letterSpacing: "normal",
                lineHeight: 1.18,
            },
            h6: {
                fontSize: "1rem",
                fontWeight: 500,
                letterSpacing: "normal",
                lineHeight: 1.2,
            },
            subtitle1: {
                fontSize: "0.875rem",
                fontWeight: 500,
                lineHeight: 1.2,
            },
            subtitle2: {
                fontSize: "0.875rem",
                fontWeight: 500,
                lineHeight: 1.2,
            },
            body1: {
                fontSize: "0.875rem",
                lineHeight: 1.43,
                fontWeight: 400,
                letterSpacing: "normal",
            },
            body2: {
                fontSize: "0.875rem",
                lineHeight: 1.43,
                fontWeight: 400,
                letterSpacing: "normal",
            },
            button: {
                fontSize: "0.875rem",
                fontWeight: 500,
                textTransform: "none",
                letterSpacing: "normal",
                lineHeight: 1.2,
            },
            overline: {
                fontSize: "0.875rem",
                fontWeight: 500,
                letterSpacing: "normal",
                textTransform: "none",
                lineHeight: 1.2,
            },
            caption: {
                fontSize: "0.875rem",
                lineHeight: 1.43,
                fontWeight: 400,
                color: colors.pewter,
            },
        },
        transitions: {
            duration: {
                shortest: motion.durationMs,
                shorter: motion.durationMs,
                short: motion.durationMs,
                standard: motion.durationMs,
                complex: motion.durationMs,
                enteringScreen: motion.durationMs,
                leavingScreen: motion.durationMs,
            },
            easing: {
                easeInOut: motion.easing,
                easeOut: motion.easing,
                easeIn: motion.easing,
                sharp: motion.easing,
            },
        },
    });

    const electricBlueHover = isDark ? "#4D78E8" : "#355FCC";

    return createTheme(theme, {
        components: {
            MuiCssBaseline: {
                styleOverrides: {
                    ":root": {
                        colorScheme: mode,
                        "--tesla-electric-blue": colors.electricBlue,
                        "--tesla-carbon-dark": colors.carbonDark,
                        "--tesla-graphite": colors.graphite,
                        "--tesla-pewter": colors.pewter,
                        "--tesla-light-ash": colors.lightAsh,
                        "--tesla-cloud-gray": colors.cloudGray,
                    },
                    "*, *::before, *::after": {
                        boxSizing: "border-box",
                    },
                    html: {
                        minHeight: "100%",
                        scrollBehavior: "smooth",
                    },
                    body: {
                        minHeight: "100vh",
                        margin: 0,
                        backgroundColor: theme.palette.background.default,
                        color: theme.palette.text.primary,
                        textRendering: "optimizeLegibility",
                        WebkitFontSmoothing: "antialiased",
                        MozOsxFontSmoothing: "grayscale",
                    },
                    "#root": {
                        minHeight: "100vh",
                    },
                    "::selection": {
                        backgroundColor: alpha(colors.electricBlue, 0.22),
                    },
                    a: {
                        color: colors.pewter,
                        textDecoration: "none",
                        transition: `box-shadow ${motion.durationMs}ms ${motion.easing}, color ${motion.durationMs}ms`,
                        "&:hover": {
                            textDecoration: "underline",
                        },
                    },
                },
            },
            MuiAppBar: {
                styleOverrides: {
                    root: {
                        backdropFilter: "none",
                        backgroundImage: "none",
                        boxShadow: "none",
                        borderBottom: "none",
                    },
                },
            },
            MuiPaper: {
                defaultProps: {
                    elevation: 0,
                },
                styleOverrides: {
                    root: {
                        backgroundImage: "none",
                        borderRadius: radii.button,
                        boxShadow: "none",
                    },
                },
            },
            MuiCard: {
                defaultProps: {
                    elevation: 0,
                },
                styleOverrides: {
                    root: {
                        borderRadius: radii.card,
                        border: "none",
                        backgroundColor: isDark ? theme.palette.background.paper : colors.white,
                        boxShadow: "none",
                    },
                },
            },
            MuiButton: {
                defaultProps: {
                    disableElevation: true,
                },
                styleOverrides: {
                    root: {
                        minHeight: 40,
                        padding: "4px 16px",
                        borderRadius: radii.button,
                        fontWeight: 500,
                        border: "3px solid transparent",
                        boxShadow: "rgba(0,0,0,0) 0px 0px 0px 2px inset",
                        transition,
                        "&:focus-visible": {
                            boxShadow: `rgba(0,0,0,0) 0px 0px 0px 2px inset, 0 0 0 2px ${alpha(colors.electricBlue, 0.45)}`,
                        },
                    },
                    contained: {
                        backgroundColor: colors.electricBlue,
                        color: colors.white,
                        "&:hover": {
                            backgroundColor: electricBlueHover,
                        },
                    },
                    containedPrimary: {
                        backgroundColor: colors.electricBlue,
                        color: colors.white,
                        "&:hover": {
                            backgroundColor: electricBlueHover,
                        },
                    },
                    outlined: {
                        backgroundColor: isDark ? "transparent" : colors.white,
                        color: isDark ? theme.palette.text.primary : colors.graphite,
                        borderColor: isDark ? alpha(colors.white, 0.16) : "transparent",
                        "&:hover": {
                            backgroundColor: isDark ? alpha(colors.white, 0.06) : colors.lightAsh,
                            borderColor: isDark ? alpha(colors.white, 0.24) : "transparent",
                        },
                    },
                    text: {
                        color: colors.pewter,
                        minHeight: 32,
                        padding: "4px 16px",
                        "&:hover": {
                            backgroundColor: isDark ? alpha(colors.white, 0.06) : colors.lightAsh,
                            textDecoration: "underline",
                        },
                    },
                    sizeSmall: {
                        minHeight: 32,
                        padding: "4px 12px",
                    },
                    sizeLarge: {
                        minHeight: 40,
                        padding: "4px 20px",
                    },
                },
            },
            MuiIconButton: {
                styleOverrides: {
                    root: {
                        borderRadius: radii.button,
                        transition,
                        "&:hover": {
                            backgroundColor: isDark ? alpha(colors.white, 0.08) : colors.lightAsh,
                        },
                    },
                },
            },
            MuiChip: {
                styleOverrides: {
                    root: {
                        borderRadius: radii.button,
                        fontWeight: 500,
                        fontSize: "0.875rem",
                    },
                    outlined: {
                        borderColor: colors.paleSilver,
                    },
                },
            },
            MuiOutlinedInput: {
                styleOverrides: {
                    root: {
                        borderRadius: radii.button,
                        backgroundColor: "transparent",
                        transition,
                        "&:hover .MuiOutlinedInput-notchedOutline": {
                            borderColor: colors.paleSilver,
                        },
                        "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                            borderColor: colors.carbonDark,
                            borderWidth: 1,
                        },
                    },
                    notchedOutline: {
                        borderColor: colors.cloudGray,
                    },
                    input: {
                        paddingBlock: 12,
                        "&::placeholder": {
                            color: colors.silverFog,
                            opacity: 1,
                        },
                    },
                },
            },
            MuiInputLabel: {
                styleOverrides: {
                    root: {
                        fontWeight: 500,
                        fontSize: "0.875rem",
                    },
                },
            },
            MuiAlert: {
                styleOverrides: {
                    root: {
                        borderRadius: radii.button,
                        boxShadow: "none",
                    },
                },
            },
            MuiAvatar: {
                styleOverrides: {
                    root: {
                        fontWeight: 500,
                    },
                },
            },
            MuiDrawer: {
                styleOverrides: {
                    paper: {
                        borderRadius: 0,
                        borderRight: `1px solid ${theme.palette.divider}`,
                        backgroundColor: theme.palette.background.paper,
                        boxShadow: "none",
                    },
                },
            },
            MuiListItemButton: {
                styleOverrides: {
                    root: {
                        borderRadius: radii.button,
                        minHeight: 40,
                        padding: "4px 16px",
                        transition,
                        "&.Mui-selected": {
                            backgroundColor: isDark ? alpha(colors.white, 0.08) : colors.lightAsh,
                            color: theme.palette.text.primary,
                            "& .MuiListItemIcon-root": {
                                color: theme.palette.text.primary,
                            },
                        },
                        "&:hover": {
                            backgroundColor: isDark ? alpha(colors.white, 0.06) : colors.lightAsh,
                        },
                    },
                },
            },
            MuiTableCell: {
                styleOverrides: {
                    head: {
                        fontWeight: 500,
                        color: theme.palette.text.secondary,
                        backgroundColor: isDark ? alpha(colors.white, 0.04) : colors.lightAsh,
                    },
                },
            },
            MuiTooltip: {
                styleOverrides: {
                    tooltip: {
                        borderRadius: radii.button,
                        backgroundColor: colors.carbonDark,
                        color: colors.white,
                        fontSize: "0.875rem",
                        boxShadow: "none",
                    },
                },
            },
            MuiDivider: {
                styleOverrides: {
                    root: {
                        borderColor: theme.palette.divider,
                    },
                },
            },
            MuiSkeleton: {
                defaultProps: {
                    animation: "wave",
                },
                styleOverrides: {
                    rounded: {
                        borderRadius: radii.button,
                    },
                },
            },
            MuiLink: {
                styleOverrides: {
                    root: {
                        color: colors.pewter,
                        fontSize: "0.875rem",
                        fontWeight: 400,
                        textDecoration: "none",
                        transition: `box-shadow ${motion.durationMs}ms ${motion.easing}, color ${motion.durationMs}ms`,
                        "&:hover": {
                            textDecoration: "underline",
                        },
                    },
                },
            },
        },
    });
}

export const lightTheme = buildTheme("light");
export const darkTheme = buildTheme("dark");

export type ColorMode = "light" | "dark" | "system";
