import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
    Alert,
    Box,
    Button,
    Collapse,
    IconButton,
    InputAdornment,
    Stack,
    TextField,
    type TextFieldProps,
    Typography,
} from "@mui/material";
import { Visibility, VisibilityOff } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { forgotPassword, signIn, signUp } from "../api/auth";
import { AuthMarketingPanel } from "../components/auth/AuthMarketingPanel";
import { AuthShell } from "../components/auth/AuthShell";
import { useAuth } from "../hooks/useAuth";
import {
    forgotPasswordSchema,
    signInSchema,
    signUpSchema,
    type ForgotPasswordValues,
    type SignInValues,
    type SignUpValues,
} from "../features/auth/schemas";
import { usePlatformMetadata } from "../hooks/usePlatformMetadata";
import { getQueryErrorMessage } from "../utils/queryErrors";

type Mode = "signIn" | "signUp";

function PasswordField(props: TextFieldProps) {
    const [showPassword, setShowPassword] = useState(false);

    return (
        <TextField
            {...props}
            type={showPassword ? "text" : "password"}
            InputProps={{
                endAdornment: (
                    <InputAdornment position="end">
                        <IconButton
                            edge="end"
                            aria-label={showPassword ? "Hide password" : "Show password"}
                            onClick={() => setShowPassword((value) => !value)}
                            onMouseDown={(event) => event.preventDefault()}
                        >
                            {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                    </InputAdornment>
                ),
            }}
        />
    );
}

function SignInForm({ onSuccess, mfaEnabled }: { onSuccess: () => void; mfaEnabled: boolean }) {
    const navigate = useNavigate();
    const { setAuthenticated } = useAuth();
    const {
        register,
        handleSubmit,
        getValues,
        formState: { errors, isSubmitting },
    } = useForm<SignInValues>({ resolver: zodResolver(signInSchema) });
    const {
        register: registerForgot,
        handleSubmit: handleForgotSubmit,
        reset: resetForgot,
        formState: {
            errors: forgotErrors,
            isSubmitting: isSubmittingForgot,
        },
    } = useForm<ForgotPasswordValues>({ resolver: zodResolver(forgotPasswordSchema) });

    const [serverError, setServerError] = useState("");
    const [forgotOpen, setForgotOpen] = useState(false);
    const [forgotDone, setForgotDone] = useState(false);
    const [forgotError, setForgotError] = useState("");

    async function onSubmit(values: SignInValues) {
        setServerError("");
        try {
            const mfaCode = values.mfa_code?.trim();
            const data = await signIn({
                ...values,
                mfa_code: mfaCode || undefined,
            });
            setAuthenticated(data.user);
            onSuccess();
            navigate("/dashboard");
        } catch (error) {
            setServerError(getQueryErrorMessage(error, "Sign in failed."));
        }
    }

    async function onForgotPasswordSubmit(values: ForgotPasswordValues) {
        setForgotError("");
        try {
            await forgotPassword(values);
            setForgotDone(true);
        } catch (error) {
            setForgotError(getQueryErrorMessage(error, "Request failed."));
        }
    }

    function toggleForgotPassword() {
        const nextVisible = !forgotOpen;
        setForgotOpen(nextVisible);
        setForgotError("");
        setForgotDone(false);
        if (nextVisible) {
            resetForgot({ email: getValues("email") });
        }
    }

    return (
        <Stack spacing={2}>
            {serverError && <Alert severity="error">{serverError}</Alert>}
            <Box component="form" onSubmit={handleSubmit(onSubmit)}>
                <Stack spacing={2}>
                    <TextField
                        label="Email"
                        type="email"
                        autoComplete="email"
                        {...register("email")}
                        error={!!errors.email}
                        helperText={errors.email?.message}
                        fullWidth
                    />
                    <PasswordField
                        label="Password"
                        autoComplete="current-password"
                        {...register("password")}
                        error={!!errors.password}
                        helperText={errors.password?.message}
                        fullWidth
                    />
                    {mfaEnabled && (
                        <TextField
                            label="Authenticator code"
                            autoComplete="one-time-code"
                            {...register("mfa_code")}
                            error={!!errors.mfa_code}
                            helperText={errors.mfa_code?.message || "Required for accounts with MFA enabled."}
                            fullWidth
                        />
                    )}
                    <Button type="submit" variant="contained" size="large" disabled={isSubmitting}>
                        {isSubmitting ? "Signing in..." : "Sign in"}
                    </Button>
                </Stack>
            </Box>
            <Button variant="text" onClick={toggleForgotPassword} sx={{ alignSelf: "flex-start", px: 0 }}>
                {forgotOpen ? "Hide password reset" : "Forgot password?"}
            </Button>
            <Collapse in={forgotOpen} unmountOnExit>
                <Stack spacing={2}>
                    <Typography variant="body2" color="text.secondary">
                        Enter your email and we will send a reset link if an account exists.
                    </Typography>
                    {forgotDone && (
                        <Alert severity="success">
                            If that email exists, a reset link has been sent. Check your inbox.
                        </Alert>
                    )}
                    {forgotError && <Alert severity="error">{forgotError}</Alert>}
                    <Box component="form" onSubmit={handleForgotSubmit(onForgotPasswordSubmit)}>
                        <Stack spacing={2}>
                            <TextField
                                label="Email"
                                type="email"
                                {...registerForgot("email")}
                                error={!!forgotErrors.email}
                                helperText={forgotErrors.email?.message}
                                fullWidth
                            />
                            <Button type="submit" variant="outlined" disabled={isSubmittingForgot}>
                                {isSubmittingForgot ? "Sending..." : "Send reset link"}
                            </Button>
                        </Stack>
                    </Box>
                </Stack>
            </Collapse>
        </Stack>
    );
}

function SignUpForm({ onSuccess }: { onSuccess: (email: string) => void }) {
    const {
        register,
        handleSubmit,
        formState: { errors, isSubmitting },
    } = useForm<SignUpValues>({ resolver: zodResolver(signUpSchema) });
    const [serverError, setServerError] = useState("");

    async function onSubmit(values: SignUpValues) {
        setServerError("");
        try {
            const adminInviteCode = values.admin_invite_code?.trim();
            await signUp({
                ...values,
                admin_invite_code: adminInviteCode || undefined,
            });
            onSuccess(values.email);
        } catch (error) {
            setServerError(getQueryErrorMessage(error, "Sign up failed."));
        }
    }

    return (
        <Stack spacing={2}>
            {serverError && <Alert severity="error">{serverError}</Alert>}
            <Box component="form" onSubmit={handleSubmit(onSubmit)}>
                <Stack spacing={2}>
                    <TextField
                        label="Full name"
                        autoComplete="name"
                        {...register("full_name")}
                        error={!!errors.full_name}
                        helperText={errors.full_name?.message}
                        fullWidth
                    />
                    <TextField
                        label="Email"
                        type="email"
                        autoComplete="email"
                        {...register("email")}
                        error={!!errors.email}
                        helperText={errors.email?.message}
                        fullWidth
                    />
                    <PasswordField
                        label="Password"
                        autoComplete="new-password"
                        {...register("password")}
                        error={!!errors.password}
                        helperText={errors.password?.message}
                        fullWidth
                    />
                    <PasswordField
                        label="Admin invite code"
                        autoComplete="one-time-code"
                        {...register("admin_invite_code")}
                        error={!!errors.admin_invite_code}
                        helperText={errors.admin_invite_code?.message || "Optional. Required only for admin registrations."}
                        fullWidth
                    />
                    <Button type="submit" variant="contained" size="large" disabled={isSubmitting}>
                        {isSubmitting ? "Creating account..." : "Create account"}
                    </Button>
                </Stack>
            </Box>
        </Stack>
    );
}

export default function AuthHomePage() {
    const { data: platformMetadata } = usePlatformMetadata();
    const [mode, setMode] = useState<Mode>("signIn");
    const [successMsg, setSuccessMsg] = useState("");

    const appName = platformMetadata?.app_name ?? "Your App";
    const coreDomainPlural = platformMetadata?.core_domain_plural ?? "projects";

    return (
        <AuthShell
            sideContent={
                <AuthMarketingPanel
                    appName={appName}
                    eyebrow="Launch-ready workspace"
                    title={`A premium way to manage your ${coreDomainPlural.toLowerCase()}.`}
                    description="The product is now framed around clarity, confidence, and modern SaaS polish, with cleaner navigation, stronger hierarchy, and better flow from first sign-in."
                    highlights={[
                        { value: "Fast", label: "Cleaner onboarding and reduced friction" },
                        { value: "Focused", label: "A calmer surface for daily operations" },
                        { value: "Secure", label: "Built around stronger trust signals" },
                    ]}
                    points={[
                        "Sign in to continue your existing workflow with a sharper interface.",
                        "Create an account to access the full platform experience and dashboard.",
                    ]}
                />
            }
        >
            <Stack spacing={3}>
                <Box>
                    <Typography variant="overline" color="text.secondary" sx={{ display: "block" }}>
                        {appName}
                    </Typography>
                    <Typography variant="h4" sx={{ mt: 0.5 }}>
                        {mode === "signIn" ? "Welcome back" : "Create your account"}
                    </Typography>
                    <Typography color="text.secondary" sx={{ mt: 1 }}>
                        {mode === "signIn"
                            ? `Sign in to manage your ${coreDomainPlural.toLowerCase()} and account activity.`
                            : `Create an account to start managing your ${coreDomainPlural.toLowerCase()} with the new workspace experience.`}
                    </Typography>
                </Box>

                <Box
                    sx={(theme) => ({
                        p: 0.5,
                        borderRadius: 1,
                        display: "grid",
                        gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                        gap: 0.5,
                        backgroundColor: theme.palette.mode === "dark" ? "action.hover" : "var(--tesla-light-ash)",
                    })}
                >
                    <Button
                        variant={mode === "signIn" ? "contained" : "text"}
                        onClick={() => {
                            setMode("signIn");
                            setSuccessMsg("");
                        }}
                    >
                        Sign in
                    </Button>
                    <Button
                        variant={mode === "signUp" ? "contained" : "text"}
                        onClick={() => {
                            setMode("signUp");
                            setSuccessMsg("");
                        }}
                    >
                        Create account
                    </Button>
                </Box>

                {successMsg && <Alert severity="success">{successMsg}</Alert>}

                {mode === "signIn" ? (
                    <SignInForm onSuccess={() => undefined} mfaEnabled={platformMetadata?.mfa_enabled ?? false} />
                ) : (
                    <SignUpForm
                        onSuccess={(email) => {
                            setSuccessMsg(`Account created for ${email}. You can sign in now.`);
                            setMode("signIn");
                        }}
                    />
                )}

                <Typography variant="body2" color="text.secondary">
                    By continuing, you are entering a cleaner product experience with improved navigation, readability, and trust cues.
                </Typography>
            </Stack>
        </AuthShell>
    );
}
