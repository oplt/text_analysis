import { createContext, useContext } from "react";
import type { ColorMode } from "./theme";

type ColorModeContextValue = {
    colorMode: ColorMode;
    setColorMode: (mode: ColorMode) => void;
};

export const ColorModeContext = createContext<ColorModeContextValue>({
    colorMode: "light",
    setColorMode: () => undefined,
});

export function useColorMode() {
    return useContext(ColorModeContext);
}
