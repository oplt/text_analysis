"""Headless CLI for reproducible text-research pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        try:
            return dict(value.__dict__)
        except Exception:
            return str(value)
    return str(value)


from backend.modules.text_research.domain.analysis_specification import AnalysisSpecification
from backend.modules.text_research.domain.prepared_corpus import compute_pipeline_checksum
from backend.modules.text_research.infrastructure.drift_monitoring import build_drift_report
from backend.modules.text_research.infrastructure.pipeline_compiler import compile_plan
from backend.modules.text_research.infrastructure.preprocessing import (
    PreprocessingConfig,
    describe_implementation,
)
from backend.modules.text_research.infrastructure.prepared_corpus_builder import prepare_texts
from backend.modules.text_research.infrastructure.text_transforms import (
    FixEncoding,
    Lowercase,
    NormalizeWhitespace,
    UnicodeNormalize,
)
from backend.modules.text_research.infrastructure.topic_engines import (
    BERTopicEngine,
    SklearnLDAEngine,
    SklearnNMFEngine,
)


def _load_json_arg(raw: str | None, path: str | None) -> dict[str, Any]:
    if raw and path:
        raise SystemExit("Provide only one of --json or --file")
    if raw:
        payload = json.loads(raw)
    elif path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        raise SystemExit("One of --json or --file is required")
    if not isinstance(payload, dict):
        raise SystemExit("Expected a JSON object")
    return payload


def cmd_compile_spec(args: argparse.Namespace) -> int:
    payload = _load_json_arg(args.json, args.file)
    spec = AnalysisSpecification.model_validate(payload)
    normalized = spec.normalize()
    normalized.validate()
    plan = compile_plan(normalized)
    output = {
        "spec": normalized.model_dump(mode="json"),
        "spec_hash": plan.spec_hash,
        "engine_version": plan.engine_version,
        "stages": plan.stages,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def cmd_fingerprint_recipe(args: argparse.Namespace) -> int:
    payload = _load_json_arg(None, args.file)
    config = PreprocessingConfig.from_dict(payload.get("preprocessing", payload))
    impl_meta = describe_implementation(config)
    transforms = payload.get("transforms")
    transform_meta = transforms if isinstance(transforms, list) else None
    if transform_meta is None:
        transform_meta = [
            step.name
            for step in (
                FixEncoding(),
                UnicodeNormalize(),
                NormalizeWhitespace(),
                Lowercase(),
            )
        ]
    fingerprint_payload = {
        "preprocessing": config.to_dict(),
        "implementation": impl_meta,
        "transforms": transform_meta,
    }
    checksum = compute_pipeline_checksum(config.to_dict(), impl_meta)
    print(
        json.dumps(
            {
                "recipe": fingerprint_payload,
                "pipeline_checksum": checksum,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_describe_plugins(_args: argparse.Namespace) -> int:
    plugins = {
        "topic_engines": [
            {"name": SklearnLDAEngine.name, "optional": False},
            {"name": SklearnNMFEngine.name, "optional": False},
            {"name": BERTopicEngine.name, "optional": True},
        ],
        "text_transforms": [
            FixEncoding().name,
            UnicodeNormalize().name,
            NormalizeWhitespace().name,
            Lowercase().name,
        ],
    }
    print(json.dumps(plugins, indent=2, sort_keys=True))
    return 0


def cmd_prepare_demo(args: argparse.Namespace) -> int:
    path = Path(args.texts_file)
    texts = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    config = PreprocessingConfig.from_dict(json.loads(args.config) if args.config else {})
    prepared = prepare_texts(texts, config)
    output = {
        "unit_count": len(texts),
        "corpus_checksum": prepared.corpus_checksum,
        "pipeline_checksum": prepared.pipeline_checksum,
        "vocabulary_size": len(prepared.vocabulary),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def cmd_check_drift(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not isinstance(current, dict):
        raise SystemExit("baseline and current JSON files must be objects")
    report = build_drift_report(baseline=baseline, current=current)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_run_analysis(args: argparse.Namespace) -> int:
    """Run an in-process StageRunner analysis from texts + spec (no DB)."""
    from backend.modules.text_research.application.analysis_executor import (
        build_spec_from_request,
        run_prepared_analysis,
    )

    texts = Path(args.texts_file).read_text(encoding="utf-8").splitlines()
    texts = [line for line in texts if line.strip()]
    if args.file:
        payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
        from backend.modules.text_research.domain.analysis_specification import (
            AnalysisSpecification,
        )

        spec = AnalysisSpecification.model_validate(payload).normalize()
        spec.validate()
    else:
        analysis_type = args.analysis_type or "frequencies"
        spec = build_spec_from_request(analysis_type, corpus_id="cli-corpus", random_seed=args.seed)
    config = json.loads(args.config) if args.config else None
    result = run_prepared_analysis(spec, texts, config=config)
    print(json.dumps(result, indent=2, ensure_ascii=True, default=_json_default))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="text-research")
    sub = parser.add_subparsers(dest="command", required=True)

    compile_spec = sub.add_parser("compile-spec", help="Normalize spec and print execution plan")
    compile_spec.add_argument("--json", help="Inline JSON specification")
    compile_spec.add_argument("--file", help="Path to JSON specification file")
    compile_spec.set_defaults(func=cmd_compile_spec)

    fingerprint = sub.add_parser("fingerprint-recipe", help="Hash a preprocessing recipe")
    fingerprint.add_argument("--file", required=True, help="Path to recipe JSON file")
    fingerprint.set_defaults(func=cmd_fingerprint_recipe)

    describe = sub.add_parser("describe-plugins", help="List optional and core plugins")
    describe.set_defaults(func=cmd_describe_plugins)

    prepare = sub.add_parser("prepare-demo", help="Prepare texts without DB access")
    prepare.add_argument("--texts-file", required=True, help="Newline-delimited texts")
    prepare.add_argument("--config", help="Optional preprocessing config JSON string")
    prepare.set_defaults(func=cmd_prepare_demo)

    drift = sub.add_parser("check-drift", help="Compare baseline/current drift JSON files")
    drift.add_argument("--baseline", required=True)
    drift.add_argument("--current", required=True)
    drift.set_defaults(func=cmd_check_drift)

    run_analysis = sub.add_parser(
        "run-analysis",
        help="Execute StageRunner on local texts (frequencies/dfm/…; no DB)",
    )
    run_analysis.add_argument("--texts-file", required=True)
    run_analysis.add_argument("--file", help="AnalysisSpecification JSON file")
    run_analysis.add_argument("--analysis-type", default="frequencies")
    run_analysis.add_argument("--config", help="Preprocessing config JSON string")
    run_analysis.add_argument("--seed", type=int, default=42)
    run_analysis.set_defaults(func=cmd_run_analysis)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SystemExit as exc:
        if exc.code is None:
            return 1
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 1
        return int(exc.code)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
