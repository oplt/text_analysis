import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";

export type UseTabQueryParamOptions<T extends string> = {
    /** Map legacy query values onto current tabs (e.g. `models` → `predictions`). */
    aliases?: Readonly<Record<string, T>>;
};

/**
 * Persist a page tab in the URL query string (`?tab=...`) so refresh/back/deep-link work.
 * Falls back to `defaultTab` when the param is missing or invalid.
 */
export function useTabQueryParam<T extends string>(
    tabs: readonly T[],
    defaultTab: T,
    paramKey = "tab",
    options?: UseTabQueryParamOptions<T>
): [T, (next: T) => void] {
    const [searchParams, setSearchParams] = useSearchParams();
    const aliases = options?.aliases;

    const value = useMemo(() => {
        const raw = searchParams.get(paramKey);
        if (!raw) return defaultTab;
        if ((tabs as readonly string[]).includes(raw)) {
            return raw as T;
        }
        const aliased = aliases?.[raw];
        if (aliased && (tabs as readonly string[]).includes(aliased)) {
            return aliased;
        }
        return defaultTab;
    }, [aliases, defaultTab, paramKey, searchParams, tabs]);

    const setValue = useCallback(
        (next: T) => {
            setSearchParams(
                (prev) => {
                    const nextParams = new URLSearchParams(prev);
                    if (next === defaultTab) {
                        nextParams.delete(paramKey);
                    } else {
                        nextParams.set(paramKey, next);
                    }
                    return nextParams;
                },
                { replace: true }
            );
        },
        [defaultTab, paramKey, setSearchParams]
    );

    return [value, setValue];
}
