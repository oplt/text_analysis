import { describe, expect, it } from "vitest";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PageTabs } from "../../components/ui/PageTabs";
import { useTabQueryParam } from "../../hooks/useTabQueryParam";

const CORPUS_TABS = ["documents", "metadata", "text_units", "quality", "import"] as const;
const RELIABILITY_TABS = [
    "overview",
    "agreement",
    "disagreements",
    "by_coder",
    "methods",
] as const;
const ANALYSIS_GROUPS = [
    "overview",
    "frequencies",
    "associations",
    "statistical",
    "measurement",
    "advanced",
] as const;
const RAG_TABS = ["documents", "indexing", "retrieval", "evaluation", "runs"] as const;
const AGENT_TABS = ["configure", "run", "trace", "sources", "output"] as const;

function TabHarness<T extends string>({
    tabs,
    defaultTab,
    aliases,
}: {
    tabs: readonly T[];
    defaultTab: T;
    aliases?: Readonly<Record<string, T>>;
}) {
    const [tab, setTab] = useTabQueryParam(tabs, defaultTab, "tab", { aliases });
    const location = useLocation();
    return (
        <div>
            <PageTabs
                value={tab}
                onChange={setTab}
                tabs={tabs.map((value) => ({ value, label: value }))}
            />
            <div data-testid="tab-value">{tab}</div>
            <div data-testid="search">{location.search || "(empty)"}</div>
        </div>
    );
}

describe("Phase 7 workspace tab contracts", () => {
    it.each([
        {
            name: "corpus",
            tabs: CORPUS_TABS,
            defaultTab: "documents" as const,
            click: "import",
            aliases: { management: "import" as const },
            aliasEntry: "?tab=management",
            aliasExpected: "import",
        },
        {
            name: "reliability",
            tabs: RELIABILITY_TABS,
            defaultTab: "overview" as const,
            click: "disagreements",
            aliases: { adjudication: "disagreements" as const },
            aliasEntry: "?tab=adjudication",
            aliasExpected: "disagreements",
        },
        {
            name: "analysis groups",
            tabs: ANALYSIS_GROUPS,
            defaultTab: "overview" as const,
            click: "frequencies",
            aliases: undefined,
            aliasEntry: undefined,
            aliasExpected: undefined,
        },
        {
            name: "rag",
            tabs: RAG_TABS,
            defaultTab: "documents" as const,
            click: "retrieval",
            aliases: undefined,
            aliasEntry: undefined,
            aliasExpected: undefined,
        },
        {
            name: "agent",
            tabs: AGENT_TABS,
            defaultTab: "configure" as const,
            click: "output",
            aliases: undefined,
            aliasEntry: undefined,
            aliasExpected: undefined,
        },
    ])("$name tabs persist in the URL", async ({ tabs, defaultTab, click, aliases }) => {
        const user = userEvent.setup();
        render(
            <MemoryRouter initialEntries={["/workspace"]}>
                <Routes>
                    <Route
                        path="/workspace"
                        element={
                            <TabHarness tabs={tabs} defaultTab={defaultTab} aliases={aliases} />
                        }
                    />
                </Routes>
            </MemoryRouter>
        );

        expect(screen.getByTestId("tab-value")).toHaveTextContent(defaultTab);
        await user.click(screen.getByRole("tab", { name: click }));
        expect(screen.getByTestId("tab-value")).toHaveTextContent(click);
        expect(screen.getByTestId("search")).toHaveTextContent(`tab=${click}`);
    });

    it("maps corpus management alias to import", () => {
        render(
            <MemoryRouter initialEntries={["/workspace?tab=management"]}>
                <Routes>
                    <Route
                        path="/workspace"
                        element={
                            <TabHarness
                                tabs={CORPUS_TABS}
                                defaultTab="documents"
                                aliases={{ management: "import" }}
                            />
                        }
                    />
                </Routes>
            </MemoryRouter>
        );
        expect(screen.getByTestId("tab-value")).toHaveTextContent("import");
    });

    it("maps reliability adjudication alias to disagreements", () => {
        render(
            <MemoryRouter initialEntries={["/workspace?tab=adjudication"]}>
                <Routes>
                    <Route
                        path="/workspace"
                        element={
                            <TabHarness
                                tabs={RELIABILITY_TABS}
                                defaultTab="overview"
                                aliases={{ adjudication: "disagreements" }}
                            />
                        }
                    />
                </Routes>
            </MemoryRouter>
        );
        expect(screen.getByTestId("tab-value")).toHaveTextContent("disagreements");
    });
});
