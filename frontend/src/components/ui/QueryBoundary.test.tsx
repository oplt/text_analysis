import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { QueryBoundary, QueryErrorAlert } from "./QueryBoundary";

describe("QueryBoundary", () => {
    it("renders children when the query succeeds", () => {
        render(
            <QueryBoundary isLoading={false} isError={false}>
                <div>Loaded content</div>
            </QueryBoundary>
        );

        expect(screen.getByText("Loaded content")).toBeInTheDocument();
    });

    it("shows an error alert with retry", async () => {
        const user = userEvent.setup();
        const onRetry = vi.fn();

        render(
            <QueryBoundary
                isError
                error={new Error("Network down")}
                errorFallback="Failed to load data."
                onRetry={onRetry}
            >
                <div>Loaded content</div>
            </QueryBoundary>
        );

        expect(screen.getByText("Network down")).toBeInTheDocument();
        await user.click(screen.getByRole("button", { name: "Retry" }));
        expect(onRetry).toHaveBeenCalledOnce();
    });

    it("renders empty fallback when requested", () => {
        render(
            <QueryBoundary isEmpty emptyFallback={<div>No records</div>}>
                <div>Loaded content</div>
            </QueryBoundary>
        );

        expect(screen.getByText("No records")).toBeInTheDocument();
    });

    it("shows a default empty message when isEmpty and no fallback", () => {
        render(
            <QueryBoundary isEmpty>
                <div>Loaded content</div>
            </QueryBoundary>
        );

        expect(screen.getByText(/Nothing to show yet/i)).toBeInTheDocument();
        expect(screen.queryByText("Loaded content")).not.toBeInTheDocument();
    });

    it("marks loading state as busy for assistive tech", () => {
        const { container } = render(
            <QueryBoundary isLoading>
                <div>Loaded content</div>
            </QueryBoundary>
        );

        expect(container.querySelector('[aria-busy="true"]')).toBeTruthy();
        expect(screen.queryByText("Loaded content")).not.toBeInTheDocument();
    });
});

describe("QueryErrorAlert", () => {
    it("returns null without an error", () => {
        const { container } = render(<QueryErrorAlert error={null} />);
        expect(container).toBeEmptyDOMElement();
    });
});
