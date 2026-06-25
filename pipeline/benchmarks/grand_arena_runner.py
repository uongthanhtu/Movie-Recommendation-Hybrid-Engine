"""
Grand Arena Runner -- config-driven CLI orchestrator for the Grand Unified Benchmark
Arena. Iterates over a selected list of datasets (or every registered dataset), loads
each via DatasetFactory, routes to the correct model set based on ArenaDataset.mode,
trains + evaluates each model, and renders a publication-ready Markdown summary table
(also saved alongside a CSV).

Per-dataset ManualDownloadRequiredError failures (e.g. Douban, which has no working
automated source -- see
docs/superpowers/specs/2026-06-24-epinions-douban-flixster-design.md) are caught and
logged as a graceful SKIP. Per-(dataset, model) failures during training or evaluation
are caught independently and logged as FAILED, so one bad combination never aborts the
rest of the sweep.

Usage:
    py -3 -m pipeline.benchmarks.grand_arena_runner --datasets filmtrust ciao ml-100k
    py -3 -m pipeline.benchmarks.grand_arena_runner --all
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pipeline.benchmarks import evaluation, model_runner
from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY, IMPLICIT_DATASET_REGISTRY
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.data_loaders.loader_utils import ManualDownloadRequiredError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("grand_arena_runner")

RESULTS_DIR = "models"
CSV_PATH = os.path.join(RESULTS_DIR, "grand_arena_results.csv")
MD_PATH = os.path.join(RESULTS_DIR, "grand_arena_results.md")


@dataclass
class _Row:
    dataset: str
    mode: str
    model: str
    status: str  # "success" | "failed"
    recall_at_10: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    train_seconds: Optional[float] = None
    latency_ms: Optional[float] = None
    note: str = ""


@dataclass
class _Results:
    rows: List[_Row] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (dataset, message)

    def record_success(self, dataset: str, mode: str, model: str, train_seconds: float, metrics: dict) -> None:
        self.rows.append(_Row(
            dataset=dataset, mode=mode, model=model, status="success",
            recall_at_10=metrics["recall@10"], ndcg_at_10=metrics["ndcg@10"],
            train_seconds=train_seconds, latency_ms=metrics["latency_ms"],
        ))

    def record_failed(self, dataset: str, mode: str, model: str, message: str) -> None:
        self.rows.append(_Row(dataset=dataset, mode=mode, model=model, status="failed", note=message))

    def record_skipped(self, dataset: str, message: str) -> None:
        self.skipped.append((dataset, message))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grand Unified Benchmark Arena orchestrator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--datasets", nargs="+", metavar="NAME", help="Dataset names, e.g. --datasets filmtrust ciao ml-100k")
    group.add_argument("--all", action="store_true", help="Run every dataset registered in either DatasetFactory registry")
    return parser.parse_args(argv)


def _resolve_dataset_names(args: argparse.Namespace) -> List[str]:
    if args.all:
        return sorted(set(DATASET_REGISTRY.keys()) | set(IMPLICIT_DATASET_REGISTRY.keys()))
    return list(args.datasets)


def _print_metadata_banner(name: str, dataset) -> None:
    density_pct = 100.0 * dataset.train_csr.nnz / (dataset.num_users * dataset.num_items)
    mode_label = "Mode A: Explicit Trust" if dataset.mode == "explicit" else "Mode B: Implicit Trust (ABLATION STUDY)"
    edge_label = "Explicit trust edges" if dataset.mode == "explicit" else "Synthetic (Jaccard) edges"
    print(f"\n{'=' * 70}")
    print(f"[{name}] {mode_label}")
    print(f"  Users={dataset.num_users:,}  Items={dataset.num_items:,}  Density={density_pct:.4f}%")
    print(f"  {edge_label}: {dataset.social_csr.nnz:,}")
    print(f"{'=' * 70}")


def _run_sweep(dataset_names: List[str], results: _Results) -> None:
    for name in dataset_names:
        try:
            dataset = DatasetFactory.create(name).load()
        except ManualDownloadRequiredError as e:
            log.warning("[%s] SKIPPED -- manual download required:\n%s", name, e)
            results.record_skipped(name, str(e))
            continue

        _print_metadata_banner(name, dataset)

        models = model_runner.MODE_A_MODELS if dataset.mode == "explicit" else model_runner.MODE_B_MODELS
        for model_name in models:
            try:
                engine, train_seconds = model_runner.run_model(model_name, dataset)
                metrics = evaluation.evaluate_model(engine, dataset, k=10)
                results.record_success(name, dataset.mode, model_name, train_seconds, metrics)
                print(
                    f"  [{name}/{model_name}] recall@10={metrics['recall@10']:.4f} "
                    f"ndcg@10={metrics['ndcg@10']:.4f} train={train_seconds:.1f}s "
                    f"latency={metrics['latency_ms']:.2f}ms"
                )
            except Exception as e:
                log.error("[%s/%s] FAILED: %s", name, model_name, e)
                results.record_failed(name, dataset.mode, model_name, str(e))


def _render_markdown(results: _Results) -> str:
    lines = ["# Grand Arena Results", ""]
    datasets_seen = sorted({row.dataset for row in results.rows})

    for dataset in datasets_seen:
        dataset_rows = [r for r in results.rows if r.dataset == dataset]
        mode_label = "Mode A: Explicit Trust" if dataset_rows[0].mode == "explicit" else "Mode B: Implicit Trust (ABLATION STUDY)"
        lines.append(f"## {dataset} -- {mode_label}")
        lines.append("")
        lines.append("| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |")
        lines.append("|---|---|---|---|---|")
        for row in dataset_rows:
            if row.status == "success":
                lines.append(
                    f"| {row.model} | {row.recall_at_10:.4f} | {row.ndcg_at_10:.4f} "
                    f"| {row.train_seconds:.1f} | {row.latency_ms:.2f} |"
                )
            else:
                lines.append(f"| {row.model} | FAILED | FAILED | FAILED | FAILED ({row.note}) |")
        lines.append("")

    if results.skipped:
        lines.append("## Skipped Datasets")
        lines.append("")
        for name, message in results.skipped:
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"```\n{message}\n```")
            lines.append("")

    return "\n".join(lines)


def _write_csv(results: _Results, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "mode", "model", "status", "recall@10", "ndcg@10", "train_seconds", "latency_ms", "note"])
        for row in results.rows:
            writer.writerow([
                row.dataset, row.mode, row.model, row.status,
                row.recall_at_10 if row.recall_at_10 is not None else "",
                row.ndcg_at_10 if row.ndcg_at_10 is not None else "",
                row.train_seconds if row.train_seconds is not None else "",
                row.latency_ms if row.latency_ms is not None else "",
                row.note,
            ])
        for name, message in results.skipped:
            writer.writerow([name, "", "", "skipped", "", "", "", "", message])


def main(argv: Optional[List[str]] = None) -> _Results:
    args = _parse_args(argv)
    dataset_names = _resolve_dataset_names(args)
    results = _Results()
    _run_sweep(dataset_names, results)

    markdown = _render_markdown(results)
    print("\n" + markdown)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    _write_csv(results, CSV_PATH)

    return results


if __name__ == "__main__":
    main()
