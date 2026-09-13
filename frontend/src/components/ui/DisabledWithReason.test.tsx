import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button, ThemeProvider, createTheme } from "@mui/material";

import { DisabledWithReason } from "./DisabledWithReason";

function wrap(ui: React.ReactElement) {
    return render(<ThemeProvider theme={createTheme()}>{ui}</ThemeProvider>);
}

describe("DisabledWithReason", () => {
    it("renders children unchanged when there is no reason", () => {
        wrap(
            <DisabledWithReason>
                <Button>Train</Button>
            </DisabledWithReason>
        );
        expect(screen.getByRole("button", { name: "Train" })).toBeEnabled();
    });

    it("exposes a tooltip explaining why a disabled control cannot run", async () => {
        const user = userEvent.setup();
        wrap(
            <DisabledWithReason reason="Select a training dataset snapshot before training.">
                <Button disabled>Train classifier</Button>
            </DisabledWithReason>
        );

        const button = screen.getByRole("button", { name: "Train classifier" });
        expect(button).toBeDisabled();
        await user.hover(button.parentElement!);
        expect(
            await screen.findByText("Select a training dataset snapshot before training.")
        ).toBeInTheDocument();
    });
});
