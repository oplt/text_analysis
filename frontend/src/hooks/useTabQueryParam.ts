import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";

/**
 * Persist a page tab in the URL query string (`?tab=...`) so refresh/back/deep-link work.
 * Falls back to `defaultTab` when the param is missing or invalid.
 */
export function useTabQueryParam<T extends string>(
    tabs: readonly T[],
    defaultTab: T,
    paramKey = "tab"
): [T, (next: T) => void] {
    const [searchParams, setSearchParams] = useSearchParams();

    const value = useMemo(() => {
        const raw = searchParams.get(paramKey);
        if (raw && (tabs as readonly string[]).includes(raw)) {
            return raw as T;
        }
        return defaultTab;
    }, [defaultTab, paramKey, searchParams, tabs]);

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
