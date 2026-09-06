.PHONY: local-dev docker-dev prod-dev observability-up observability-down observability-logs observability-status start-observability-stack stop-observability-stack local-dev-no-observability fix check install-hooks commit-ready

local-dev:
	$(MAKE) -f Makefile.local local-dev

docker-dev:
	$(MAKE) -f Makefile.docker docker-dev

prod-dev:
	$(MAKE) -f Makefile.deploy prod-dev

observability-up:
	$(MAKE) -f Makefile.local observability-up

observability-down:
	$(MAKE) -f Makefile.local observability-down

observability-logs:
	$(MAKE) -f Makefile.local observability-logs

observability-status start-observability-stack stop-observability-stack local-dev-no-observability:
	$(MAKE) -f Makefile.local $@

fix:
	ruff check . --fix
	ruff format .
	cd frontend && npx biome check --write .

check:
	ruff check .
	ruff format --check .
	cd frontend && npx biome check .
	cd frontend && npx tsc --noEmit

install-hooks:
	pre-commit install

commit-ready: fix check
