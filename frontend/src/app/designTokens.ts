/** Tesla-inspired design tokens from DESIGN.md */
export const colors = {
    electricBlue: "#3E6AE1",
    white: "#FFFFFF",
    lightAsh: "#F4F4F4",
    carbonDark: "#171A20",
    graphite: "#393C41",
    pewter: "#5C5E62",
    silverFog: "#8E8E8E",
    cloudGray: "#EEEEEE",
    paleSilver: "#D0D1D2",
    frostedGlass: "rgba(255, 255, 255, 0.75)",
    overlay: "rgba(128, 128, 128, 0.65)",
} as const;

export const fonts = {
    display: "'Universal Sans Display', -apple-system, Arial, sans-serif",
    text: "'Universal Sans Text', -apple-system, Arial, sans-serif",
} as const;

export const motion = {
    durationMs: 330,
    easing: "cubic-bezier(0.5, 0, 0, 0.75)",
} as const;

export const radii = {
    button: 4,
    card: 12,
    none: 0,
} as const;
