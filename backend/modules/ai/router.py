from fastapi import APIRouter

from backend.modules.ai.document_routes import router as document_router
from backend.modules.ai.evaluation_routes import router as evaluation_router
from backend.modules.ai.overview_routes import router as overview_router
from backend.modules.ai.prompt_routes import router as prompt_router
from backend.modules.ai.review_routes import router as review_router
from backend.modules.ai.run_routes import router as run_router

router = APIRouter()
router.include_router(overview_router)
router.include_router(prompt_router)
router.include_router(document_router)
router.include_router(run_router)
router.include_router(review_router)
router.include_router(evaluation_router)
