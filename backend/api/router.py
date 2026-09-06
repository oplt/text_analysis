from fastapi import APIRouter

from backend.modules.admin.router import router as admin_router
from backend.modules.ai.agent_router import router as agent_router
from backend.modules.ai.router import router as ai_router
from backend.modules.calendar.router import router as calendar_router
from backend.modules.identity_access.router import router as auth_router
from backend.modules.memory.api.routes import router as memory_router
from backend.modules.notifications.router import router as notifications_router
from backend.modules.platform.router import router as platform_router
from backend.modules.profile.router import router as profile_router
from backend.modules.projects.router import router as projects_router
from backend.modules.rag.api.routes import router as rag_router
from backend.modules.settings.router import router as settings_router
from backend.modules.users.router import router as users_router
from backend.observability.router import router as observability_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(ai_router, prefix="/ai", tags=["ai"])
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(memory_router, prefix="/memory", tags=["memory"])
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(calendar_router, prefix="/calendar", tags=["calendar"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(profile_router, prefix="/profile", tags=["profile"])
api_router.include_router(projects_router, prefix="/projects", tags=["projects"])
api_router.include_router(notifications_router, prefix="/notifications", tags=["notifications"])
api_router.include_router(observability_router, prefix="/observability", tags=["observability"])
api_router.include_router(platform_router, prefix="/platform", tags=["platform"])
api_router.include_router(settings_router, prefix="/settings", tags=["settings"])
api_router.include_router(admin_router, prefix="/admin", tags=["admin"])
