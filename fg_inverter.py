from __future__ import annotations
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

import os
import random
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rdkit import Chem as _Chem
from rdkit.Chem import MolToSmiles
from rdkit.Chem.Draw import MolToImage

from fraginv.fg_core import (
    FGCandidate,
    MotifLibrary,
    capacity_loss,
    round_to_atomic,
    safe_logp_from_smiles,
)
from fraginv.utils import GraphFromMol
from fraginv.train import add_extra_features


def seed_everything(seed: int, deterministic: bool = False) -> None:
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_model_input_width(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    first_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        return x

    need = int(first_linear.in_features)

    if x.dim() == 2:
        cur = x.shape[-1]
        if cur == need:
            return x
        if cur < need:
            return torch.cat([x, x.new_zeros(x.shape[0], need - cur)], dim=-1)
        return x[:, :need]

    if x.dim() == 3:
        b, n, f = x.shape
        if f == need:
            return x
        if f < need:
            return torch.cat([x, x.new_zeros(b, n, need - f)], dim=-1)
        return x[..., :need]

    return x


def maybe_denormalize_logp(pred: torch.Tensor, config: Any) -> torch.Tensor:
    y = pred.view(-1)

    mean = None
    std = None
    if hasattr(config, "logp_mean") and hasattr(config, "logp_std"):
        mean = float(config.logp_mean)
        std = float(config.logp_std)
    elif hasattr(config, "y_mean") and hasattr(config, "y_std"):
        mean = float(config.y_mean)
        std = float(config.y_std)

    if mean is None or std is None:
        return y
    return y * std + mean


def logp_range_loss_raw(
    pred_logp_raw: torch.Tensor,
    lo: float,
    hi: float,
    hi_mult: float = 10.0,
    lo_mult: float = 1.0,
) -> torch.Tensor:
    y = pred_logp_raw.view(-1)
    above = F.relu(y - hi)
    below = F.relu(lo - y)
    return (hi_mult * above.pow(2)) + (lo_mult * below.pow(2))


def _perturb_init(fg_var: FGCandidate, scale: float) -> None:
    if scale <= 0:
        return
    with torch.no_grad():
        fg_var.S.add_(scale * torch.randn_like(fg_var.S))
        fg_var.A.add_(scale * torch.randn_like(fg_var.A))


def _enforce_fg_constraints(fg_var: FGCandidate) -> None:
    with torch.no_grad():
        upper = torch.triu(fg_var.A, 1).clone()
        sym = upper + upper.transpose(0, 1)
        fg_var.A.copy_(sym)
        fg_var.A.fill_diagonal_(float("-inf"))
        fg_var.S.clamp_(-8.0, 8.0)


def _ensure_fg_defaults(config: Any, lib: MotifLibrary) -> SimpleNamespace:
    if not hasattr(config, "fg") or not isinstance(config.fg, SimpleNamespace):
        setattr(config, "fg", SimpleNamespace())
    fg = config.fg

    fg.steps       = int(getattr(fg, "steps", 300))
    fg.print_every = int(getattr(fg, "print_every", 25))

    fg.max_motifs  = int(getattr(fg, "max_motifs", 32))
    fg.max_atoms   = int(getattr(fg, "max_atoms", int(getattr(config, "max_size", 80))))
    fg.min_atoms   = int(getattr(fg, "min_atoms", 15))

    fg.init_scale  = float(getattr(fg, "init_scale", 0.02))
    fg.init_noise  = float(getattr(fg, "init_noise", 0.25))

    # NES
    fg.sigma       = float(getattr(fg, "sigma", 0.10))
    fg.pop         = int(getattr(fg, "pop", 16))
    fg.lr_es       = float(getattr(fg, "lr_es", 0.20))

    # weights
    fg.w_prop      = float(getattr(fg, "w_prop", 1.0))
    fg.w_cap       = float(getattr(fg, "w_cap", 0.05))
    fg.w_size      = float(getattr(fg, "w_size", 0.20))
    fg.w_invalid   = float(getattr(fg, "w_invalid", 5.0))

    fg.early_stop_patience  = int(getattr(fg, "early_stop_patience", 80))
    fg.early_stop_min_delta = float(getattr(fg, "early_stop_min_delta", 1e-4))
    fg.early_stop_start     = int(getattr(fg, "early_stop_start", fg.warmup_valid_steps))

    if hasattr(fg, "logp_lo") and hasattr(fg, "logp_hi"):
        fg.logp_lo = float(getattr(fg, "logp_lo"))
        fg.logp_hi = float(getattr(fg, "logp_hi"))
    else:
        target = getattr(config, "target", None)
        tol = float(getattr(fg, "logp_tol", 0.25))
        if target is None:
            # safe fallback
            fg.logp_lo = float(getattr(fg, "logp_lo_fallback", -0.5))
            fg.logp_hi = float(getattr(fg, "logp_hi_fallback", 2.0))
        else:
            t = float(target)
            fg.logp_lo = t - tol
            fg.logp_hi = t + tol

    fg.hi_mult     = float(getattr(fg, "hi_mult", 10.0))
    fg.lo_mult     = float(getattr(fg, "lo_mult", 1.0))

    fg.warmup_valid_steps = int(getattr(fg, "warmup_valid_steps", max(30, int(fg.steps * 0.15))))
    fg.invalid_patience   = int(getattr(fg, "invalid_patience", 25))
    fg.w_prop_warm = float(getattr(fg, "w_prop_warm", 0.0))
    fg.w_cap_warm  = float(getattr(fg, "w_cap_warm", max(1.0, 10.0 * float(getattr(fg, "w_cap", 0.05)))))

    fg.sigma_min = float(getattr(fg, "sigma_min", 0.03))
    fg.lr_min    = float(getattr(fg, "lr_min", 0.05))
    return fg


@torch.no_grad()
def evaluate_candidate(
    fg_var: FGCandidate,
    model: nn.Module,
    lib: MotifLibrary,
    config: Any,
    fg_cfg: SimpleNamespace,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Optional[_Chem.Mol], Optional[str]]:

    mol = None
    smi = None
    valid = 0.0
    n_atoms = 0

    try:
        mol = round_to_atomic(fg_var)
        if mol is not None:
            n_atoms = int(mol.GetNumAtoms())
            _Chem.SanitizeMol(mol)
            smi = MolToSmiles(mol)
            valid = 1.0
    except Exception:
        valid = 0.0
        smi = None
        mol = None
        n_atoms = 0

    if valid > 0.0 and mol is not None and n_atoms > 0:
        try:
            fea, adj = GraphFromMol(mol, N=max(1, n_atoms))
            fea = fea.to(device)
            adj = adj.to(device)

            atom_fea_ext = add_extra_features(fea, config._extra_fea_matrix).unsqueeze(0)
            atom_fea_ext = ensure_model_input_width(atom_fea_ext, model)

            pred = model(atom_fea_ext, adj.unsqueeze(0))
            pred_logp_raw = maybe_denormalize_logp(pred, config)

            L_prop = logp_range_loss_raw(
                pred_logp_raw,
                lo=fg_cfg.logp_lo,
                hi=fg_cfg.logp_hi,
                hi_mult=fg_cfg.hi_mult,
                lo_mult=fg_cfg.lo_mult,
            ).mean()

            pred_logp_val = float(pred_logp_raw.mean().cpu())
        except Exception:
            L_prop = torch.tensor(10.0, device=device)
            pred_logp_val = float("nan")
    else:
        L_prop = torch.tensor(10.0, device=device)
        pred_logp_val = float("nan")

    L_cap_raw = capacity_loss(fg_var, w=1.0)
    excess = max(0, n_atoms - int(fg_cfg.max_atoms))
    deficit = max(0, int(fg_cfg.min_atoms) - n_atoms)
    size_term = (excess / 20.0) ** 2 + (deficit / 10.0) ** 2
    L_size = torch.tensor(float(size_term), device=device)

    L_invalid = torch.tensor(float(0.0 if valid > 0.0 else 1.0), device=device)

    total = (
        fg_cfg.w_prop * L_prop
        + fg_cfg.w_cap * L_cap_raw
    )
    info = {
        "TOTAL": float(total.cpu()),
        "L_prop": float((fg_cfg.w_prop * L_prop).cpu()),
        "L_cap": float((fg_cfg.w_cap * L_cap_raw).cpu()),
        "L_size": float((fg_cfg.w_size * L_size).cpu()),
        "L_invalid": float((fg_cfg.w_invalid * L_invalid).cpu()),
        "pred_logP": float(pred_logp_val),
        "atoms": float(n_atoms),
        "valid": float(valid),
    }
    return float(total.cpu()), info, mol, smi


def nes_step(
    fg_var: FGCandidate,
    model: nn.Module,
    lib: MotifLibrary,
    config: Any,
    fg_cfg: SimpleNamespace,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Optional[_Chem.Mol], Optional[str]]:

    pop = int(fg_cfg.pop)
    sigma = float(fg_cfg.sigma)
    lr = float(fg_cfg.lr_es)

    with torch.no_grad():
        base_S = fg_var.S.detach().clone()
        base_A = fg_var.A.detach().clone()

    eps_S = torch.randn((pop,) + fg_var.S.shape, device=device)
    eps_A = torch.randn((pop,) + fg_var.A.shape, device=device)

    losses: List[float] = []
    infos: List[Dict[str, float]] = []
    mols: List[Optional[_Chem.Mol]] = []
    smis: List[Optional[str]] = []

    for i in range(pop):
        with torch.no_grad():
            fg_var.S.copy_(base_S + sigma * eps_S[i])
            fg_var.A.copy_(base_A + sigma * eps_A[i])
            _enforce_fg_constraints(fg_var)

        L, info, mol, smi = evaluate_candidate(fg_var, model, lib, config, fg_cfg, device)
        losses.append(L)
        infos.append(info)
        mols.append(mol)
        smis.append(smi)

    losses_t = torch.tensor(losses, device=device, dtype=torch.float32)
    normed = (losses_t - losses_t.mean()) / (losses_t.std() + 1e-8)

    gS = (normed.view(pop, *([1] * fg_var.S.dim())) * eps_S).mean(dim=0) / sigma
    gA = (normed.view(pop, *([1] * fg_var.A.dim())) * eps_A).mean(dim=0) / sigma

    with torch.no_grad():
        fg_var.S.copy_(base_S - lr * gS)
        fg_var.A.copy_(base_A - lr * gA)
        _enforce_fg_constraints(fg_var)

    best_i = int(torch.argmin(losses_t).item())
    return losses[best_i], infos[best_i], mols[best_i], smis[best_i]


def run_fg_inversion(
    model: nn.Module,
    lib: MotifLibrary,
    config: Any,
    output: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    fg_cfg = _ensure_fg_defaults(config, lib)

    if seed is not None:
        seed_everything(seed, deterministic=False)

    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "drawings"), exist_ok=True)

    # freeze predictor
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    fg_cfg.w_size = 0.0 #5.0
    fg_cfg.w_invalid = 0.0 #0.1

    w_prop_final = float(getattr(fg_cfg, "w_prop", 1.0))
    w_cap_final  = float(getattr(fg_cfg, "w_cap", 0.05))

    warmup_valid_steps = int(fg_cfg.warmup_valid_steps)
    invalid_patience   = int(fg_cfg.invalid_patience)

    w_prop_warm = float(getattr(fg_cfg, "w_prop_warm", 0.0))
    w_cap_warm  = float(getattr(fg_cfg, "w_cap_warm", max(1.0, 10.0 * w_cap_final)))

    sigma_min = float(getattr(fg_cfg, "sigma_min", 0.03))
    lr_min    = float(getattr(fg_cfg, "lr_min", 0.05))

    # init candidate
    fg_var = FGCandidate(int(fg_cfg.max_motifs), lib, init_scale=float(fg_cfg.init_scale), device=device)
    _perturb_init(fg_var, float(fg_cfg.init_noise))
    _enforce_fg_constraints(fg_var)

    best_any = {"loss": float("inf"), "smiles": None, "pred_logP": None, "rdkit_logP": None, "mol": None, "info": None}
    best_valid = {"loss": float("inf"), "smiles": None, "pred_logP": None, "rdkit_logP": None, "mol": None, "info": None}
    best_in_range = {"loss": float("inf"), "smiles": None, "pred_logP": None, "rdkit_logP": None, "mol": None, "info": None}
    best_score = float("inf")
    no_improve = 0

    early_patience  = int(getattr(fg_cfg, "early_stop_patience", 0))
    early_min_delta = float(getattr(fg_cfg, "early_stop_min_delta", 0.0))
    early_start     = int(getattr(fg_cfg, "early_stop_start", 0))

    mid = 0.5 * (float(fg_cfg.logp_lo) + float(fg_cfg.logp_hi))

    def _patience_score(L_total: float, info: Dict[str, float]) -> float:
        """Lower is better."""
        valid = (info.get("valid", 0.0) > 0.0)
        pred = info.get("pred_logP", float("nan"))

        if (not valid) or pred is None or (isinstance(pred, float) and np.isnan(pred)):
            return 1e9 + float(L_total)

        pred = float(pred)
        in_range = (fg_cfg.logp_lo <= pred <= fg_cfg.logp_hi)
        if in_range:
            return float(L_total)
        return abs(pred - mid) + 0.01 * float(L_total)

    def _update_best(L: float, info: Dict[str, float], mol: Optional[_Chem.Mol], smi: str, rdkit_lp: float) -> None:
        nonlocal best_any, best_valid, best_in_range

        if L < best_any["loss"]:
            best_any = {"loss": float(L), "smiles": smi, "pred_logP": info.get("pred_logP"), "rdkit_logP": rdkit_lp, "mol": mol, "info": info}

        is_valid = (info.get("valid", 0.0) > 0.0 and smi != "<smiles_failed>")
        if is_valid and L < best_valid["loss"]:
            best_valid = {"loss": float(L), "smiles": smi, "pred_logP": info.get("pred_logP"), "rdkit_logP": rdkit_lp, "mol": mol, "info": info}

        in_range = is_valid and (info.get("pred_logP") is not None) and (fg_cfg.logp_lo <= info["pred_logP"] <= fg_cfg.logp_hi)
        if in_range and L < best_in_range["loss"]:
            best_in_range = {"loss": float(L), "smiles": smi, "pred_logP": info.get("pred_logP"), "rdkit_logP": rdkit_lp, "mol": mol, "info": info}

    # start in feasibility mode
    fg_cfg.w_prop = w_prop_warm
    fg_cfg.w_cap  = w_cap_warm

    invalid_streak = 0
    found_any_valid = False

    for step in range(int(fg_cfg.steps)):
        # curriculum
        if (not found_any_valid) and step < warmup_valid_steps:
            fg_cfg.w_prop = w_prop_warm
            fg_cfg.w_cap  = w_cap_warm
        else:
            if found_any_valid:
                t = (step - warmup_valid_steps) / max(1.0, (fg_cfg.steps - warmup_valid_steps))
                t = float(np.clip(t, 0.0, 1.0))
                fg_cfg.w_prop = (1.0 - t) * w_prop_warm + t * w_prop_final
                fg_cfg.w_cap  = (1.0 - t) * w_cap_warm  + t * w_cap_final
            else:
                fg_cfg.w_prop = w_prop_warm
                fg_cfg.w_cap  = w_cap_warm

        # NES step
        L, info, mol, _ = nes_step(fg_var, model, lib, config, fg_cfg, device)

        # robust SMILES
        smi_print = "<smiles_failed>"
        if mol is not None:
            try:
                _Chem.SanitizeMol(mol)
                smi_print = _Chem.MolToSmiles(mol)
            except Exception:
                smi_print = "<smiles_failed>"

        rdkit_lp = safe_logp_from_smiles(smi_print, default=float("nan"))

        # always print
        print(
            f"[FG] step={step:4d} smi={smi_print} "
            f"pred_logP={info.get('pred_logP')} rdkit_logP={rdkit_lp}",
            flush=True,
        )

        _update_best(L, info, mol, smi_print, rdkit_lp)

        # ---- early stopping (patience) ----
        if early_patience > 0 and step >= early_start:
            s = _patience_score(L, info)
            if s < best_score - early_min_delta:
                best_score = s
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_patience:
                    print(
                        f"[FG] early stop: no improvement for {early_patience} steps "
                        f"(best_score={best_score:.6f})",
                        flush=True,
                    )
                    break

        is_valid = (info.get("valid", 0.0) > 0.0 and smi_print != "<smiles_failed>")
        if is_valid:
            found_any_valid = True
            invalid_streak = 0
        else:
            invalid_streak += 1

        if invalid_streak >= invalid_patience:
            fg_cfg.sigma = max(sigma_min, float(fg_cfg.sigma) * 0.7)
            fg_cfg.lr_es = max(lr_min, float(fg_cfg.lr_es) * 0.7)
            fg_cfg.w_cap = float(fg_cfg.w_cap) * 1.25
            invalid_streak = 0

    chosen = best_in_range if best_in_range["mol"] is not None else (best_valid if best_valid["mol"] is not None else best_any)
    final_mol = chosen["mol"]
    final_smi = chosen["smiles"]

    try:
        if final_mol is not None:
            MolToImage(final_mol).save(os.path.join(output, "drawings", "fg_result.png"))
    except Exception:
        pass

    return {
        "smiles": final_smi,
        "pred_logP": chosen["pred_logP"],
        "rdkit_logP": chosen["rdkit_logP"],
        "loss": float(chosen["loss"]),
        "n_iter": int(fg_cfg.steps),
        "best_info": chosen["info"],
    }


