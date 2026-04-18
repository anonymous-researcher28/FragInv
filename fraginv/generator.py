
from __future__ import annotations

import os
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from rdkit import Chem

from .config import default_config_dict
from .train import train
from fraginv.fg_core import MotifLibrary
from fraginv.fg_inverter import run_fg_inversion

try:
    from fraginv.motif_resources import MotifResources
except Exception:
    MotifResources = None


def _seed_all(seed: int) -> None:
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cfg_dir(config_path: Optional[str]) -> Path:
    if config_path is None:
        return Path(".").resolve()
    return Path(config_path).expanduser().resolve().parent


def _resolve_path(p: Any, base: Path) -> Any:
    if p is None:
        return None
    if isinstance(p, str):
        s = p.strip()
        if s == "":
            return p
        if ("/" in s) or ("\\" in s) or s.startswith(".") or s.startswith("~"):
            ps = Path(s).expanduser()
            return str((base / ps).resolve()) if not ps.is_absolute() else str(ps)
        return p
    return p


def to_SimpleNamespace(conf_dict: Dict[str, Any]) -> SimpleNamespace:
    def _rec(v):
        if isinstance(v, dict):
            return SimpleNamespace(**{kk: _rec(vv) for kk, vv in v.items()})
        return v
    return _rec(conf_dict)


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge b into a (returns new dict)."""
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def get_extra_features_matrix(type_list: List[str], extra_features: List[str], device: torch.device) -> torch.Tensor:
    pt = Chem.GetPeriodicTable()
    cols = []
    extra_features = extra_features or []

    if "atomic_weight" in extra_features:
        cols.append([pt.GetAtomicWeight(t) for t in type_list])
    if "atomic_number" in extra_features:
        cols.append([pt.GetAtomicNumber(t) for t in type_list])
    if "n_valence" in extra_features:
        cols.append([pt.GetNOuterElecs(t) / 8 for t in type_list])

    if len(cols) == 0:
        ef = np.zeros((len(type_list), 0), dtype=np.float32)
    else:
        ef = np.stack(cols, axis=1).astype(np.float32)

    return torch.from_numpy(ef).to(device)


def _load_config(config: Optional[str]) -> Tuple[SimpleNamespace, Path]:
    base = _cfg_dir(config)

    if config is None:
        config_dict: Dict[str, Any] = {}
    elif isinstance(config, dict):
        config_dict = config
    else:
        with open(config, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader) or {}

    # resolve some known path fields
    for key in ["ckpt_path", "resources_root", "analysis_root"]:
        if key in config_dict:
            config_dict[key] = _resolve_path(config_dict[key], base)

    if isinstance(config_dict.get("fg", None), dict):
        for key in ["vocab_dir", "edges_file"]:
            if key in config_dict["fg"]:
                config_dict["fg"][key] = _resolve_path(config_dict["fg"][key], base)
        ports = config_dict["fg"].get("ports", None)
        if isinstance(ports, dict) and "file" in ports:
            ports["file"] = _resolve_path(ports["file"], base)

    merged = _deep_merge(default_config_dict, config_dict)
    cfg = to_SimpleNamespace(merged)

    if not hasattr(cfg, "fg") or cfg.fg is None:
        cfg.fg = SimpleNamespace()

    return cfg, base


def _pick_best_fg_result(results: List[Dict[str, Any]], target: float, tol: float = 1e9) -> Optional[Dict[str, Any]]:
    """
    Prefer:
      1) in-range (if fg.logp_lo/hi exists) or |pred-target| <= tol
      2) else smallest |pred-target|
      3) else smallest loss
    """
    best = None
    best_score = float("inf")

    for r in results:
        if not r or r.get("smiles") in (None, "<smiles_failed>"):
            continue

        pred = r.get("pred_logP", None)
        loss = float(r.get("loss", 1e18))

        if pred is not None and not (isinstance(pred, float) and np.isnan(pred)):
            score = abs(float(pred) - float(target))
        else:
            score = loss

        if score < best_score:
            best_score = score
            best = r

    return best


def generate(n: int, output: str, config=None, seed: Optional[int] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, _base = _load_config(config)

    if not hasattr(cfg, "type_list") or not isinstance(cfg.type_list, list) or len(cfg.type_list) == 0:
        raise RuntimeError("Config error: type_list is empty (bad YAML load/merge).")

    base_seed = int(seed) if seed is not None else int(getattr(cfg, "seed", 0) or 0)
    if base_seed == 0 and seed is None and getattr(cfg, "seed", None) is None:
        base_seed = int.from_bytes(os.urandom(4), "little")
    _seed_all(base_seed)

    # matrices expected by train()/model
    cfg.bonding = torch.tensor(cfg.bonding, device=device)
    cfg.bonding_mask = [(cfg.bonding == i) for i in set(cfg.bonding.tolist()) if i > 0]
    cfg._extra_fea_matrix = get_extra_features_matrix(cfg.type_list, getattr(cfg, "extra_features", []), device)

    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "drawings"), exist_ok=True)

    motif_res = None
    if MotifResources is not None:
        res_root = getattr(cfg, "resources_root", None)
        if res_root is not None:
            motif_res = MotifResources(res_root)

    model = train(cfg, output, motif_res=motif_res)
    print("Starting molecule generation loop", flush=True)

    if getattr(cfg, "generation_mode", "didgen") != "fg":
        raise RuntimeError("This generator.py is cleaned for FG mode. Set generation_mode: fg.")

    # ---- FG mode ----
    vocab_dir = getattr(cfg.fg, "vocab_dir", None)
    if vocab_dir is None:
        raise RuntimeError("Config error: fg.vocab_dir is missing")

    # attachments file (use fg.ports.file if present)
    attachments_file = "attachments.json"
    if hasattr(cfg.fg, "ports") and cfg.fg.ports is not None and hasattr(cfg.fg.ports, "file"):
        attachments_file = getattr(cfg.fg.ports, "file")  # may be full path

    lib = MotifLibrary(vocab_dir=vocab_dir, device=device, attachments_file=attachments_file)
    lib.prior_beta = 2.0          # bigger = stronger bias
    lib.prior_alpha = 0.5         # <1 flatten; 1 uses raw counts
    lib.prior_eps = 1.0
    lib.prior_default_count = 1.0

    n_restarts = int(getattr(cfg, "n_restarts", 1))
    target = float(getattr(cfg, "target", 0.0))

    all_best: List[Dict[str, Any]] = []

    for i in range(int(n)):
        per_restart: List[Dict[str, Any]] = []

        for r in range(n_restarts):
            run_seed = base_seed + (i + 1) * 100_000 + r * 1_000
            _seed_all(run_seed)

            run_out = os.path.join(output, f"mol_{i:04d}", f"restart_{r:02d}")
            os.makedirs(run_out, exist_ok=True)

            res = run_fg_inversion(model, lib, cfg, run_out, seed=run_seed)
            if isinstance(res, dict):
                res["seed"] = int(run_seed)
                res["mol_index"] = int(i)
                res["restart"] = int(r)
                per_restart.append(res)

        best = _pick_best_fg_result(per_restart, target=target)
        if best is None:
            print(f"[FG] FAILED mol {i} (no valid result)", flush=True)
            continue

        all_best.append(best)
        print(
            f"[FG] DONE mol {i}: seed={best.get('seed')} restart={best.get('restart')} "
            f"loss={best.get('loss')} pred_logP={best.get('pred_logP')} smiles={best.get('smiles')}",
            flush=True,
        )

    out_txt = os.path.join(output, "fg_results.tsv")
    with open(out_txt, "w") as f:
        f.write("mol_index\trestart\tseed\tloss\tpred_logP\trdkit_logP\tsmiles\n")
        for r in all_best:
            f.write(
                f"{r.get('mol_index')}\t{r.get('restart')}\t{r.get('seed')}\t"
                f"{r.get('loss')}\t{r.get('pred_logP')}\t{r.get('rdkit_logP')}\t{r.get('smiles')}\n"
            )

    return all_best
