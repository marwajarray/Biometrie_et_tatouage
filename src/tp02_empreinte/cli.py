from __future__ import annotations

import argparse
import json
from pathlib import Path

from .method_ssim_edges import similarity as ssim_similarity, verify as ssim_verify, SSIMConfig
from .method_orb import similarity as orb_similarity, verify as orb_verify, ORBConfig
from .method_fft import similarity as fft_similarity, verify as fft_verify, FFTConfig
from .method_gabor import similarity as gabor_similarity, verify as gabor_verify, GaborConfig
from .benchmark import load_pairs_csv, run_benchmark


def _method_dispatch(method: str):
    m = method.lower().strip()
    if m == "ssim":
        return (SSIMConfig, ssim_similarity, ssim_verify)
    if m == "orb":
        return (ORBConfig, orb_similarity, orb_verify)
    if m == "fft":
        return (FFTConfig, fft_similarity, fft_verify)
    if m == "gabor":
        return (GaborConfig, gabor_similarity, gabor_verify)
    raise ValueError("Méthode inconnue. Choix: ssim, orb, fft, gabor")


def cmd_verify(args: argparse.Namespace) -> int:
    Cfg, _, verify_fn = _method_dispatch(args.method)
    cfg = Cfg()  # defaults

    # override threshold si fourni
    if args.threshold is not None:
        cfg = cfg.__class__(**{**cfg.__dict__, "decision_threshold": float(args.threshold)})

    dec = verify_fn(args.ref, args.probe, cfg)
    out = {
        "method": args.method,
        "ref": args.ref,
        "probe": args.probe,
        "score": dec.score,
        "threshold": cfg.decision_threshold,
        "accepted": bool(dec.accepted),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    results = {}

    dec = ssim_verify(args.ref, args.probe, SSIMConfig())
    results["ssim"] = {"score": dec.score, "accepted": dec.accepted}

    dec = orb_verify(args.ref, args.probe, ORBConfig())
    results["orb"] = {"score": dec.score, "accepted": dec.accepted}

    dec = fft_verify(args.ref, args.probe, FFTConfig())
    results["fft"] = {"score": dec.score, "accepted": dec.accepted}

    dec = gabor_verify(args.ref, args.probe, GaborConfig())
    results["gabor"] = {"score": dec.score, "accepted": dec.accepted}

    print(json.dumps({"ref": args.ref, "probe": args.probe, "results": results}, indent=2, ensure_ascii=False))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    pairs = load_pairs_csv(args.pairs)
    res = run_benchmark(pairs, method=args.method, threshold=args.threshold)
    print(json.dumps(res.__dict__, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tp02_empreinte", description="TP02 - Reconnaissance d’empreinte digitale")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="Vérifier une empreinte probe contre ref")
    p_verify.add_argument("--method", required=True, choices=["ssim", "orb", "fft", "gabor"])
    p_verify.add_argument("--ref", required=True, type=str)
    p_verify.add_argument("--probe", required=True, type=str)
    p_verify.add_argument("--threshold", type=float, default=None)
    p_verify.set_defaults(func=cmd_verify)

    p_cmp = sub.add_parser("compare", help="Comparer toutes les méthodes (scores + décisions)")
    p_cmp.add_argument("--ref", required=True, type=str)
    p_cmp.add_argument("--probe", required=True, type=str)
    p_cmp.set_defaults(func=cmd_compare)

    p_bench = sub.add_parser("benchmark", help="Benchmark sur un CSV de paires")
    p_bench.add_argument("--pairs", required=True, type=str, help="CSV avec colonnes: ref,probe,label")
    p_bench.add_argument("--method", required=True, choices=["ssim", "orb", "fft", "gabor"])
    p_bench.add_argument("--threshold", type=float, default=None)
    p_bench.set_defaults(func=cmd_benchmark)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
