import { z } from "zod";
export const accountSchema = z.object({ full_name: z.string().optional() });
export const profileSchema = z.object({ bio: z.string().max(1000, "Bio is too long").nullable().optional(), location: z.string().max(255, "Location is too long").nullable().optional(), website: z.string().max(500, "Website is too long").nullable().optional() });
export const passwordSchema = z.object({ current_password: z.string().min(1, "Required"), new_password: z.string().min(8, "At least 8 characters"), confirm_password: z.string() }).refine((data) => data.new_password === data.confirm_password, { message: "Passwords do not match", path: ["confirm_password"] });
export const mfaCodeSchema = z.object({ code: z.string().regex(/^\d{6}$/, "Enter the 6-digit authenticator code") });
export type AccountValues = z.infer<typeof accountSchema>; export type ProfileValues = z.infer<typeof profileSchema>; export type PasswordValues = z.infer<typeof passwordSchema>; export type MfaCodeValues = z.infer<typeof mfaCodeSchema>;
