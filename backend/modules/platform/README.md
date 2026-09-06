# Platform module

## Ownership

- `router.py`: public router aggregator only.
- `config_routes.py` / `config_service.py`: metadata, module packs, defaults, admin config.
- `billing_routes.py` / `billing_service.py`: plans and user subscriptions.
- `api_key_routes.py` / `api_key_service.py`: user API-key lifecycle.
- `webhook_routes.py` / `webhook_service.py`: webhook lifecycle, validation, delivery tests.
- `feature_flag_routes.py` / `feature_flag_service.py`: admin flags and effective user rollout.
- `email_template_routes.py` / `email_template_service.py`: template CRUD and rendering.
- `service.py`: compatibility facade used by startup and narrow read adapters.

Response mappers remain in their owning route file. Promote one to a capability serializer only
after another module needs the exact same public mapping.

## Compatibility guarantees

The aggregator registers 24 method/path pairs. Capability splits must preserve dependencies,
schemas, status codes, error text, audit timing, transaction ownership, and cache invalidation.
`backend/tests/test_platform_cache.py` detects missing or duplicate route registrations.
