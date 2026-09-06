import { apiFetch } from "./client";

export type Profile = {
    user_id: string;
    bio: string | null;
    avatar_url: string | null;
    location: string | null;
    website: string | null;
};

export async function getProfile(): Promise<Profile> {
    return apiFetch("/profile");
}

export async function updateProfile(payload: {
    bio?: string | null;
    location?: string | null;
    website?: string | null;
}): Promise<Profile> {
    return apiFetch("/profile", {
        method: "PUT",
        body: JSON.stringify(payload),
    });
}

export async function uploadAvatar(file: File): Promise<Profile> {
    const formData = new FormData();
    formData.append("file", file);
    return apiFetch("/profile/avatar", {
        method: "POST",
        body: formData,
    });
}

export async function deleteAvatar(): Promise<void> {
    return apiFetch("/profile/avatar", {
        method: "DELETE",
    });
}
