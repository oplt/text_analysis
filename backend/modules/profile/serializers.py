from backend.modules.profile.models import UserProfile
from backend.modules.profile.schemas import ProfileResponse


def profile_to_response(profile: UserProfile) -> ProfileResponse:
    return ProfileResponse(
        user_id=profile.user_id,
        bio=profile.bio,
        avatar_url=(
            "/api/v1/profile/avatar/content" if profile.avatar_storage_key else profile.avatar_url
        ),
        location=profile.location,
        website=profile.website,
    )
