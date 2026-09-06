import { useEffect, useState } from "react";
import { Alert, Box, Button, CircularProgress, Stack, TextField, Typography } from "@mui/material";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import { resendVerification, verifyEmail } from "../api/auth";
import { AuthMarketingPanel } from "../components/auth/AuthMarketingPanel";
import { AuthShell } from "../components/auth/AuthShell";
import { resendVerificationSchema } from "../features/auth/schemas";
import { usePlatformMetadata } from "../hooks/usePlatformMetadata";
import { getQueryErrorMessage } from "../utils/queryErrors";

export default function VerifyEmailPage() {
    const [searchParams] = useSearchParams();
    const token = searchParams.get("token");
    const [email, setEmail] = useState(searchParams.get("email") ?? "");
    const [status, setStatus] = useState<"loading" | "success" | "error" | "no-token">(
        token ? "loading" : "no-token"
    );
    const [resendDone, setResendDone] = useState(false);
    const [resending, setResending] = useState(false);
    const [resendError, setResendError] = useState("");
    const { data: platformMetadata } = usePlatformMetadata();
    const appName = platformMetadata?.app_name ?? "Your App";

    useEffect(() => {
        if (!token) {
            return;
        }
        verifyEmail({ token })
            .then(() => setStatus("success"))
            .catch(() => setStatus("error"));
    }, [token]);

    async function handleResend() {
        const parsed = resendVerificationSchema.safeParse({ email: email.trim() });
        if (!parsed.success) {
            setResendError(parsed.error.issues[0]?.message ?? "Enter a valid email address.");
            return;
        }

        setResending(true);
        setResendError("");
        try {
            await resendVerification({ email: parsed.data.email });
            setResendDone(true);
        } catch (error) {
            setResendError(getQueryErrorMessage(error, "Failed to resend verification email."));
        } finally {
            setResending(false);
        }
    }

    return (
        <AuthShell
            sideContent={
                <AuthMarketingPanel
                    appName={appName}
                    eyebrow="Identity confirmation"
                    title="Verify your email and unlock the full experience."
                    description="The verification flow is clearer now, with direct guidance when links expire and a simpler resend path when users need another attempt."
                    highlights={[
                        { value: "Trust", label: "Better account legitimacy signals" },
                        { value: "Clarity", label: "More obvious next steps" },
                        { value: "Speed", label: "A simpler resend flow" },
                    ]}
                    points={[
                        "Use the original verification link if it is still valid.",
                        "If it expired, request another message without leaving the page.",
                    ]}
                />
            }
        >
            <Stack spacing={3}>
                <Box>
                    <Typography variant="overline" color="primary.main">
                        Email verification
                    </Typography>
                    <Typography variant="h4" sx={{ mt: 0.5 }}>
                        Confirm your email address
                    </Typography>
                    <Typography color="text.secondary" sx={{ mt: 1 }}>
                        Verification improves account trust and unlocks the full sign-in experience.
                    </Typography>
                </Box>

                {status === "loading" && (
                    <Box sx={{ display: "grid", placeItems: "center", py: 4 }}>
                        <CircularProgress />
                    </Box>
                )}

                {status === "success" && (
                    <Stack spacing={2}>
                        <Alert severity="success">Your email has been verified. You can now sign in.</Alert>
                        <Button component={RouterLink} to="/" variant="contained">
                            Go to sign in
                        </Button>
                    </Stack>
                )}

                {(status === "error" || status === "no-token") && (
                    <Stack spacing={2}>
                        {status === "error" && (
                            <Alert severity="error">
                                The verification link is invalid or has expired.
                            </Alert>
                        )}
                        {status === "no-token" && (
                            <Alert severity="info">
                                Check your email for a verification link, or request a new one below.
                            </Alert>
                        )}
                        <TextField
                            label="Email"
                            type="email"
                            value={email}
                            onChange={(event) => {
                                setEmail(event.target.value);
                                setResendError("");
                            }}
                            error={Boolean(resendError) && !resendDone}
                            helperText={
                                resendError && !resendDone ? resendError : "Use the same address you registered with."
                            }
                            fullWidth
                        />
                        {resendDone && <Alert severity="success">A new verification email has been sent.</Alert>}
                        {!resendDone && (
                            <Button variant="contained" disabled={resending} onClick={handleResend}>
                                {resending ? "Sending..." : "Resend verification email"}
                            </Button>
                        )}
                        <Button component={RouterLink} to="/" variant="text">
                            Back to sign in
                        </Button>
                    </Stack>
                )}
            </Stack>
        </AuthShell>
    );
}
