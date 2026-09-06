from fastapi import APIRouter

from backend.modules.platform.api_key_routes import router as api_key_router
from backend.modules.platform.billing_routes import router as billing_router
from backend.modules.platform.config_routes import router as config_router
from backend.modules.platform.email_template_routes import router as email_template_router
from backend.modules.platform.feature_flag_routes import router as feature_flag_router
from backend.modules.platform.webhook_routes import router as webhook_router

router = APIRouter()
router.include_router(config_router)
router.include_router(billing_router)
router.include_router(api_key_router)
router.include_router(webhook_router)
router.include_router(feature_flag_router)
router.include_router(email_template_router)
