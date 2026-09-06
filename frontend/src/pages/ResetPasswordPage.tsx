import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Alert, Box, Button, Stack, TextField, Typography } from "@mui/material";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import { resetPassword } from "../api/auth";
import { AuthMarketingPanel } from "../components/auth/AuthMarketingPanel";
import { AuthShell } from "../components/auth/AuthShell";
import { usePlatformMetadata } from "../hooks/usePlatformMetadata";
import { getQueryErrorMessage } from "../utils/queryErrors";

const schema = z
    .object({
        password: z.string().min(8, "At least 8 characters"),
        confirm_password: z.string(),
    })
    .refine((data) => data.password === data.confirm_password, {
        message: "Passwords do not match",
        path: ["confirm_password"],
    });

type Values = z.infer<typeof schema>;

export default function ResetPasswordPage() {
    const [searchParams] = useSearchParams();
    const token = searchParams.get("token") ?? "";
    const [done, setDone] = useState(false);
    const [serverError, setServerError] = useState("");
    const { data: platformMetadata } = usePlatformMetadata();
    const appName = platformMetadata?.app_name ?? "Your App";

    const {
        register,
        handleSubmit,
        formState: { errors, isSubmitting },
    } = useForm<Values>({ resolver: zodResolver(schema) });

    async function onSubmit(values: Values) {
        if (!token) {
            setServerError("Invalid or missing reset token.");
            return;
        }
        setServerError("");
        try {
            await resetPassword({ token, new_password: values.password });
            setDone(true);
        } catch (error) {
            setServerError(getQueryErrorMessage(error, "Reset failed."));
        }
    }

    return (
        <AuthShell
            sideContent={
                <AuthMarketingPanel
                    appName={appName}
                    eyebrow="Account recovery"
                    title="Reset access with confidence."
                    description="The recovery flow now matches the rest of the product: calmer layout, clearer feedback, and less ambiguity about what happens next."
                    highlights={[
                        { value: "1", label: "Secure token-based reset" },
                        { value: "2", label: "Clear validation feedback" },
                        { value: "3", label: "Fast path back to sign-in" },
                    ]}
                    points={[
                        "Choose a new password that is strong and memorable.",
                        "Once complete, you can return directly to sign in.",
                    ]}
                />
            }
        >
            <Stack spacing={3}>
                <Box>
                    <Typography variant="overline" color="primary.main">
                        Reset password
                    </Typography>
                    <Typography variant="h4" sx={{ mt: 0.5 }}>
                        Choose a new password
                    </Typography>
                    <Typography color="text.secondary" sx={{ mt: 1 }}>
                        Use a secure password that you have not used elsewhere.
                    </Typography>
                </Box>

                {done ? (
                    <Stack spacing={2}>
                        <Alert severity="success">Password reset successfully.</Alert>
                        <Button component={RouterLink} to="/" variant="contained">
                            Return to sign in
                        </Button>
                    </Stack>
                ) : (
                    <Box component="form" onSubmit={handleSubmit(onSubmit)}>
                        <Stack spacing={2}>
                            {serverError && <Alert severity="error">{serverError}</Alert>}
                            {!token && (
                                <Alert severity="warning">
                                    No reset token found. Please use the link from your email.
                                </Alert>
                            )}
                            <TextField
                                label="New password"
                                type="password"
                                {...register("password")}
                                error={!!errors.password}
                                helperText={errors.password?.message}
                                fullWidth
                            />
                            <TextField
                                label="Confirm new password"
                                type="password"
                                {...register("confirm_password")}
                                error={!!errors.confirm_password}
                                helperText={errors.confirm_password?.message}
                                fullWidth
                            />
                            <Button type="submit" variant="contained" disabled={isSubmitting || !token}>
                                {isSubmitting ? "Resetting..." : "Reset password"}
                            </Button>
                        </Stack>
                    </Box>
                )}
            </Stack>
        </AuthShell>
    );
}
