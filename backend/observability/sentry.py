import logging

logger = logging.getLogger(__name__)


def setup_sentry(dsn: str, environment: str, traces_sample_rate: float) -> None:
    if not dsn:
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            traces_sample_rate=traces_sample_rate,
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
        )
        logger.info("Sentry initialised (env=%s)", environment)
    except ImportError:
        logger.warning("sentry-sdk not installed - Sentry disabled")
