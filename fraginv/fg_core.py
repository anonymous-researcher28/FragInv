
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import torch
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem import SanitizeFlags
from rdkit.Chem import Crippen


def safe_mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    """Return RDKit Mol or None (never throws)."""
    if not smi:
        return None
    try:
        return Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        return None


def safe_logp_from_smiles(smi: str, default=None):
    mol = safe_mol_from_smiles(smi)
    if mol is None:
        return default
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return default


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """
    Return the largest connected component (safe).
    """
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return mol

    if not frags:
        return mol

    big = max(frags, key=lambda m: m.GetNumAtoms())
    big = Chem.Mol(big)
    try:
        Chem.SanitizeMol(big)
    except Exception:
        try:
            Chem.SanitizeMol(
                big,
                sanitizeOps=(
                    SanitizeFlags.SANITIZE_SYMMRINGS
                    | SanitizeFlags.SANITIZE_KEKULIZE
                    | SanitizeFlags.SANITIZE_SETAROMATICITY
                    | SanitizeFlags.SANITIZE_ADJUSTHS
                ),
            )
        except Exception:
            pass
    return big

MOTIF_COUNTS_LIST = [
    ("C=C", 927312),
    ("N", 376752),
    ("C=O", 251510),
    ("C", 332363),
    ("C=N", 162838),
    ("CC", 144257),
    ("S", 61100),
    ("CCC", 54901),
    ("[NH4+]", 50911),
    ("F", 71441),
    ("Cl", 38593),
    ("CCCC", 26623),
    ("[OH-]", 21835),
    ("CC(C)C", 14120),
    ("C#N", 12242),
    ("Br", 11484),
    ("C1CC1", 8242),
    ("N=N", 6080),
    ("C1CCCC1", 5105),
    ("C=S", 3990),
    ("CC1CC1", 2719),
    ("O=S", 2002),
    ("C#C", 1787),
    ("[NH2-]", 1365),
    ("CCC1CC1", 1055),
    ("C1CCC1", 776),
    ("I", 788),
    ("CC1CCC1", 576),
    ("[SH-]", 397),
    ("CC1(C)CC1", 200),
    ("CC1CC1C", 115),
]

class MotifLibrary:
    def __init__(
        self,
        vocab_dir: str,
        device: Union[str, torch.device] = "cpu",
        attachments_file: str = "attachments.json",
    ):
        self.device = torch.device(device)

        # Resolve attachments path
        att_path = attachments_file
        if not os.path.isabs(att_path):
            # treat as relative to vocab_dir
            att_path = os.path.join(vocab_dir, attachments_file)

        if not os.path.exists(att_path):
            raise FileNotFoundError(f"MotifLibrary: attachments file not found: {att_path}")

        with open(att_path, "r") as f:
            self.attachments: Dict[str, List[List[int]]] = json.load(f)

        # -----------------------------
        # OPTIONAL: filter motifs by logP
        # -----------------------------
        disable = os.environ.get("MOTIF_LOGP_DISABLE", "0").strip() == "1"
        lo_env = os.environ.get("MOTIF_LOGP_LO", None)
        hi_env = os.environ.get("MOTIF_LOGP_HI", None)

        if (not disable) and (lo_env is not None) and (hi_env is not None):
            lo = float(lo_env)
            hi = float(hi_env)
            kept: Dict[str, List[List[int]]] = {}
            dropped = 0
            invalid = 0
            for smi, cfgs in self.attachments.items():
                lp = safe_logp_from_smiles(smi, default=None)
                if lp is None:
                    invalid += 1
                    continue
                if lo <= lp <= hi:
                    kept[smi] = cfgs
                else:
                    dropped += 1
            self.attachments = kept
            print(
                f"[MotifLibrary] logP filter ON lo={lo} hi={hi} "
                f"kept={len(kept)} dropped={dropped} invalid={invalid}",
                flush=True,
            )
        else:
            print("[MotifLibrary] logP filter OFF", flush=True)

        # canonical motif list (post-filter)
        all_smiles = sorted(self.attachments.keys())
        self.smiles_to_idx = {s: i for i, s in enumerate(all_smiles)}
        self.idx_to_smiles = {i: s for s, i in self.smiles_to_idx.items()}
        self.T = len(all_smiles)

        self.prior_alpha = float(getattr(self, "prior_alpha", 0.5))
        self.prior_eps = float(getattr(self, "prior_eps", 1.0))
        self.prior_default_count = float(getattr(self, "prior_default_count", 1.0))

        # Map from motif -> count
        counts_map = {s: float(self.prior_default_count) for s in all_smiles}
        for smi, cnt in MOTIF_COUNTS_LIST:
            if smi in counts_map:
                counts_map[smi] = float(cnt)

        c = torch.tensor([counts_map[s] for s in all_smiles], device=self.device, dtype=torch.float32)
        p = (c + self.prior_eps).pow(self.prior_alpha)
        p = p / (p.sum() + 1e-12)

        # Store for later use
        self.prior_p = p
        self.prior_logp = torch.log(p + 1e-12)

        # Debug print (optional)
        try:
            topk = min(10, self.T)
            vals, idxs = torch.topk(self.prior_p, k=topk)
            top = [(all_smiles[int(i)], float(v)) for v, i in zip(vals.cpu(), idxs.cpu())]
            print(f"[MotifLibrary] prior ready (T={self.T}) alpha={self.prior_alpha} eps={self.prior_eps}. Top: {top}", flush=True)
        except Exception:
            print(f"[MotifLibrary] prior ready (T={self.T})", flush=True)
        max_ports_simul = []
        max_ports_union = []
        self.union_ports_map: Dict[str, List[int]] = {}

        for smi in all_smiles:
            cfgs = self.attachments.get(smi, [])
            simul = max((len(cfg) for cfg in cfgs), default=0)
            ports_union = sorted({p for cfg in cfgs for p in cfg})
            max_ports_simul.append(simul)
            max_ports_union.append(len(ports_union))
            self.union_ports_map[smi] = ports_union

        self.max_ports_simul = torch.tensor(max_ports_simul, dtype=torch.float32, device=self.device)
        self.max_ports_union = torch.tensor(max_ports_union, dtype=torch.float32, device=self.device)

        # choose UNION by default (better for chainables like C=C)
        self.capacity_mode = "union"  # or "simul"
        self.min_cap_floor = 2.0

    def _load_scaled_prior_weights(self, prior_csv: str, col: str, delta: float):
        """
        Build mild weights in [1-delta, 1+delta] using log(count+1) scaling.
        Motifs missing from CSV get weight 1.0.
        """
        if (not prior_csv) or (not os.path.exists(prior_csv)) or delta <= 0:
            self._motif_prior_w = {}
            return

        counts = {}
        with open(prior_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m = row.get("motif_smiles", "").strip()
                if not m:
                    continue
                try:
                    v = float(row.get(col, ""))
                except Exception:
                    v = float("nan")
                if (not math.isfinite(v)) or v <= 0:
                    continue
                counts[m] = v

        xs = []
        for m in getattr(self, "motifs", []):
            v = counts.get(m)
            if v is not None:
                xs.append(math.log(v + 1.0))

        if not xs:
            self._motif_prior_w = {}
            return

        x_min, x_max = min(xs), max(xs)
        denom = (x_max - x_min) if (x_max > x_min) else 1.0

        w = {}
        for m in getattr(self, "motifs", []):
            v = counts.get(m)
            if v is None:
                w[m] = 1.0
                continue
            x = math.log(v + 1.0)
            s = (x - x_min) / denom          # in [0,1]
            w[m] = 1.0 + delta * (2.0 * s - 1.0)  # in [1-delta, 1+delta]

        self._motif_prior_w = w


def _scaled_occurrence_log_bias(
        all_smiles: List[str],
        prior_csv: str,
        col: str = "total_occurrences",
        delta: float = 0.20,
    ) -> List[float]:

        if (not prior_csv) or (not os.path.exists(prior_csv)) or delta <= 0:
            return [0.0 for _ in all_smiles]

        counts: Dict[str, float] = {}
        with open(prior_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = (row.get("motif_smiles") or "").strip()
                if not smi:
                    continue
                try:
                    v = float(row.get(col, ""))
                except Exception:
                    continue
                if not math.isfinite(v) or v <= 0:
                    continue
                counts[smi] = v
        xs = []
        for smi in all_smiles:
            if smi in counts:
                xs.append(math.log(counts[smi] + 1.0))

        if not xs:
            return [0.0 for _ in all_smiles]

        x_min, x_max = min(xs), max(xs)
        denom = (x_max - x_min) if (x_max > x_min) else 1.0

        biases = []
        for smi in all_smiles:
            v = counts.get(smi, None)
            if v is None:
                biases.append(0.0)
                continue

            x = math.log(v + 1.0)
            s = (x - x_min) / denom          # [0,1]
            w = 1.0 + delta * (2.0 * s - 1.0)  # [1-delta, 1+delta]
            biases.append(math.log(w))

        return biases

class FGCandidate:
    def __init__(self, N: int, lib: MotifLibrary, init_scale: float = 0.01, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self.N = N
        self.lib = lib
        prior_beta = float(getattr(lib, "prior_beta", 2.0))  # strength of prior at init

        noise = torch.randn(N, lib.T, device=self.device) * init_scale

        if hasattr(lib, "prior_logp") and lib.prior_logp is not None and prior_beta > 0:
            prior_bias = prior_beta * lib.prior_logp.unsqueeze(0)  # (1,T) -> (N,T)
            S0 = noise + prior_bias
        else:
            S0 = noise

        self.S = S0.requires_grad_(True)
        A0 = (torch.randn(N, N, device=self.device) * init_scale)
        A0 = torch.tril(A0, diagonal=-1)
        A0 = A0 + A0.T
        self.A = A0.requires_grad_(True)

    def softmax_S(self) -> torch.Tensor:
        return torch.softmax(self.S, dim=1)

    def squashed_A(self) -> torch.Tensor:
        return torch.sigmoid(self.A)

    def expected_ports(self) -> torch.Tensor:
        Sprob = self.softmax_S()  # (N,T)
        caps = self.lib.max_ports_simul if getattr(self.lib, "capacity_mode", "union") == "simul" else self.lib.max_ports_union
        exp_cap = (Sprob * caps.unsqueeze(0)).sum(dim=1)  # (N,)
        floor = float(getattr(self.lib, "min_cap_floor", 2.0))
        return torch.maximum(exp_cap, torch.full_like(exp_cap, floor))


def capacity_loss(fg: FGCandidate, w: float = 1.0) -> torch.Tensor:
    A = fg.squashed_A()
    deg = A.sum(dim=1)
    cap = fg.expected_ports()
    over = torch.relu(deg - cap)
    return w * (over.pow(2)).sum()


def st_hard(x_soft: torch.Tensor, dim: Optional[int] = None, thresh: float = 0.5) -> torch.Tensor:
    if dim is None:
        y_hard = (x_soft > thresh).float()
    else:
        y_hard = torch.zeros_like(x_soft)
        idx = torch.argmax(x_soft, dim=dim)
        y_hard.scatter_(dim, idx.unsqueeze(dim), 1.0)
    return y_hard - x_soft.detach() + x_soft


_MAX_SINGLE_BOND_DEG = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def _would_exceed_single_bond_cap(rw: RWMol, atom_idx: int) -> bool:
    a = rw.GetAtomWithIdx(atom_idx)
    Z = a.GetAtomicNum()
    cap = _MAX_SINGLE_BOND_DEG.get(Z, 4)
    return (a.GetDegree() + 1) > cap


def round_to_atomic(fg: FGCandidate) -> Chem.Mol:
    Sprob = fg.softmax_S()
    Aprob = fg.squashed_A()

    # hard motif choice per slot
    S_hat = st_hard(Sprob, dim=1)
    # hard adjacency
    A_hat = st_hard(torch.triu(Aprob, 1), dim=None, thresh=0.5)
    A_hat = A_hat + A_hat.T

    rw = RWMol()
    node_to_atom_indices: List[List[int]] = []
    chosen_ports_per_node: List[List[int]] = []

    # 1) instantiate motifs into global mol
    for i in range(fg.N):
        t = int(torch.argmax(S_hat[i]).item())
        smi = fg.lib.idx_to_smiles[t]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            node_to_atom_indices.append([])
            chosen_ports_per_node.append([])
            continue

        local_to_global = []
        for a in m.GetAtoms():
            local_to_global.append(rw.AddAtom(Chem.Atom(a.GetAtomicNum())))
        for b in m.GetBonds():
            i0 = local_to_global[b.GetBeginAtomIdx()]
            i1 = local_to_global[b.GetEndAtomIdx()]
            try:
                rw.AddBond(i0, i1, b.GetBondType())
            except Exception:
                pass

        node_to_atom_indices.append(local_to_global)
        chosen_ports_per_node.append([])
    deg_need = A_hat.sum(dim=1).to(dtype=torch.long).tolist()
    for i in range(fg.N):
        need = int(deg_need[i])
        if need <= 0:
            chosen_ports_per_node[i] = []
            continue

        t = int(torch.argmax(S_hat[i]).item())
        smi = fg.lib.idx_to_smiles[t]
        
        cfgs = fg.lib.attachments.get(smi, [])
        cfgs_sorted = sorted(cfgs, key=lambda c: len(c), reverse=True)

        picked = None
        for cfg in cfgs_sorted:
            if len(cfg) >= need:
                picked = cfg[:need]
                break

        if picked is not None:
            chosen_ports_per_node[i] = list(picked)
        else:
            ports_union = list(fg.lib.union_ports_map.get(smi, []))
            if len(ports_union) == 0:
                chosen_ports_per_node[i] = []
            elif len(ports_union) >= need:
                chosen_ports_per_node[i] = ports_union[:need]
            else:
                k = len(ports_union)
                reuse = need - k
                chosen_ports_per_node[i] = ports_union + ports_union[-1:] * reuse
    used_counts = [Counter() for _ in range(fg.N)]

    def _next_port(node_idx: int) -> Optional[int]:
        ports = chosen_ports_per_node[node_idx]
        if not ports:
            return None
        for p in ports:
            if used_counts[node_idx][p] == 0:
                return p
        return min(ports, key=lambda x: used_counts[node_idx][x])

    for i in range(fg.N):
        for j in range(i + 1, fg.N):
            if A_hat[i, j] < 0.5:
                continue

            pi = _next_port(i)
            pj = _next_port(j)
            if pi is None or pj is None:
                continue

            if pi >= len(node_to_atom_indices[i]) or pj >= len(node_to_atom_indices[j]):
                continue

            gi = node_to_atom_indices[i][pi]
            gj = node_to_atom_indices[j][pj]

            if _would_exceed_single_bond_cap(rw, gi) or _would_exceed_single_bond_cap(rw, gj):
                used_counts[i][pi] += 1
                used_counts[j][pj] += 1
                continue

            try:
                rw.AddBond(gi, gj, Chem.rdchem.BondType.SINGLE)
                used_counts[i][pi] += 1
                used_counts[j][pj] += 1
            except Exception:
                pass

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    return _largest_fragment(mol)
