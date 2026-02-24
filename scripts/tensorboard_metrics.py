from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    HISTOGRAMS,
    IMAGES,
    SCALARS,
)


def _normalize_path(log_path: str | Path) -> Path:
    path = Path(log_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Log path not found: {path}")
    return path


def _default_size_guidance() -> Dict[int, int]:
    # 0 means load all events for that type
    return {SCALARS: 0, HISTOGRAMS: 0, IMAGES: 0}


def _keep_n_steps(events: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    """
    Keep ~n evenly spaced events, always including the first and last.
    Assumes 'events' are ordered by step (TensorBoard typically returns ordered lists).
    """
    if n <= 0:
        return []
    m = len(events)
    if m <= n:
        return events

    # Evenly spaced indices from 0 to m-1, inclusive.
    idxs = {round(i * (m - 1) / (n - 1)) for i in range(n)}
    return [events[i] for i in sorted(idxs)]

def load_tensorboard_metrics(
    log_path: str | Path,
    size_guidance: Optional[Dict[int, int]] = None,
    keep_steps_per_metric: int = 5,
    ignore_patterns:  Optional[List[str]] = None, # <-- NOUVEAU PARAMÈTRE
) -> Dict[str, Any]:
    
    # Par défaut, on ignore rien si la liste n'est pas fournie
    if ignore_patterns is None:
        ignore_patterns = []

    path = _normalize_path(log_path)
    accumulator = EventAccumulator(str(path), size_guidance=size_guidance or _default_size_guidance())
    accumulator.Reload()

    # Fonction utilitaire pour vérifier si un tag doit être ignoré
    def should_ignore(tag_name: str) -> bool:
        for pattern in ignore_patterns:
            if pattern in tag_name:
                return True
        return False

    scalars: Dict[str, Iterable[Any]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        if should_ignore(tag): # <-- FILTRAGE ICI
            continue
            
        events = [
            {"step": e.step, "wall_time": e.wall_time, "value": e.value}
            for e in accumulator.Scalars(tag)
        ]
        scalars[tag] = _keep_n_steps(events, keep_steps_per_metric)

    histograms: Dict[str, Iterable[Any]] = {}
    for tag in accumulator.Tags().get("histograms", []):
        if should_ignore(tag): # <-- FILTRAGE ICI
            continue

        events = [
            {
                "step": e.step,
                "wall_time": e.wall_time,
                "min": e.histogram_value.min,
                "max": e.histogram_value.max,
                "num": e.histogram_value.num,
                "sum": e.histogram_value.sum,
                "sum_squares": e.histogram_value.sum_squares,
                "bucket_limit": list(e.histogram_value.bucket_limit),
                "bucket": list(e.histogram_value.bucket),
            }
            for e in accumulator.Histograms(tag)
        ]
        histograms[tag] = _keep_n_steps(events, keep_steps_per_metric)

    images: Dict[str, Iterable[Any]] = {}
    for tag in accumulator.Tags().get("images", []):
        if should_ignore(tag): # <-- FILTRAGE ICI
            continue

        events = [
            {
                "step": e.step,
                "wall_time": e.wall_time,
                "width": e.width,
                "height": e.height,
                "colorspace": getattr(e, "colorspace", None),
                "encoded_image": base64.b64encode(e.encoded_image_string).decode("ascii"),
            }
            for e in accumulator.Images(tag)
        ]
        images[tag] = _keep_n_steps(events, keep_steps_per_metric)

    return {
        "path": str(path),
        "scalars": scalars,
        "histograms": histograms,
        "images": images,
    }

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract scalar, histogram, and image data from TensorBoard event logs."
    )
    parser.add_argument(
        "log_path",
        help="Path to a TensorBoard log directory or an events.out.tfevents* file.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="tensorboard_metrics.json",
        help="Output JSON file path (default: tensorboard_metrics.json).",
    )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Do not print JSON to stdout (only write the file).",
    )
    parser.add_argument(
        "--keep-steps",
        type=int,
        default=5,
        help="Keep at most N steps per metric/tag (default: 5).",
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        default=[], 
        help="Liste de mots-clés pour ignorer certains tags (ex: --ignore train val loss).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ignored = args.ignore

    data = load_tensorboard_metrics(
        args.log_path, 
        keep_steps_per_metric=args.keep_steps,
        ignore_patterns=ignored  # <-- Passer la liste ici
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.pretty:
        text = json.dumps(data, indent=2, sort_keys=True)
    else:
        text = json.dumps(data)

    out_path.write_text(text, encoding="utf-8")

    if not args.no_stdout:
        print(text)


if __name__ == "__main__":
    main()
