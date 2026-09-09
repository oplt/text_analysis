.PHONY: local-dev docker-dev prod-dev db-migrate observability-up observability-down observability-logs observability-status start-observability-stack stop-observability-stack local-dev-no-observability fix check install-hooks commit-ready test-backend test-frontend bench-1k bench-10k ci-local backend-sync backend-lock backend-lock-check

# Deterministic backend deps (Phase 14): install from committed backend/uv.lock
backend-sync:
	cd backend && ./scripts/sync-deps.sh

backend-lock:
	cd backend && uv lock
	cd backend && uv export --frozen --no-dev --no-emit-project -o requirements.txt

backend-lock-check:
	cd backend && uv lock --check

db-migrate:
	$(MAKE) -f Makefile.local db-migrate

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
	ruff check backend --fix
	ruff format backend
	cd frontend && npm run lint -- --fix || true

check:
	ruff check backend
	ruff format --check backend
	cd frontend && npm run lint
	cd frontend && npx tsc --noEmit

test-backend:
	PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
		backend/.venv/bin/pytest backend/modules/text_research/tests -q \
		--ignore=backend/modules/text_research/tests/test_benchmark_scale.py

test-frontend:
	cd frontend && npm test

bench-1k:
	PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 BENCHMARK_INCLUDE_10K=0 BENCHMARK_INCLUDE_100K=0 \
		backend/.venv/bin/pytest backend/modules/text_research/tests/test_benchmark_scale.py::ScaleBenchmarkTests::test_benchmark_1k_units -q -s

bench-10k:
	PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 BENCHMARK_INCLUDE_10K=1 BENCHMARK_INCLUDE_100K=0 \
		backend/.venv/bin/pytest backend/modules/text_research/tests/test_benchmark_scale.py::ScaleBenchmarkTests::test_benchmark_10k_units -q -s

bench-workflows:
	PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 BENCHMARK_INCLUDE_10K=0 BENCHMARK_INCLUDE_100K=0 BENCHMARK_INCLUDE_1M=0 \
		backend/.venv/bin/pytest backend/modules/text_research/tests/test_benchmark_scale.py::WorkflowBenchmarkTests -q -s

bench-all:
	PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 BENCHMARK_INCLUDE_10K=1 BENCHMARK_INCLUDE_100K=0 BENCHMARK_INCLUDE_1M=0 \
		backend/.venv/bin/pytest backend/modules/text_research/tests/test_benchmark_scale.py -q -s

ci-local: check test-backend test-frontend bench-1k bench-workflows
	@echo "Local quality gates passed for $$(git rev-parse HEAD)"

install-hooks:
	pre-commit install

commit-ready: fix check
