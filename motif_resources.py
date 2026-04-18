
from pathlib import Path
import json, pandas as pd, numpy as np

class MotifResources:
    def __init__(self, root: str):
        root = Path(root)
        self.vocab_dir = root / "vocab"
        self.bonds_dir = root / "bond_orders"
        self.redund_dir = root / "redundancy"

        with open(self.vocab_dir / "attachments.json") as f:
            self.attachments = json.load(f)

        eo = pd.read_csv(self.bonds_dir / "edge_orders.csv")
        eo["count"] = eo[["single","double","triple","aromatic"]].sum(axis=1).astype(float)
        eo = eo[eo["count"] > 0].copy()
        eo["key_a"] = eo["motif_a"].astype(str)
        eo["key_b"] = eo["motif_b"].astype(str)
        self.pair_counts = eo[["key_a","key_b","count"]].copy()

        total = float(self.pair_counts["count"].sum())
        self.pair_counts["prior"] = self.pair_counts["count"] / max(total, 1.0)

        motifs = pd.read_csv(self.redund_dir / "motifs_redundancy.csv")
        motifs["motif_smiles"] = motifs["motif_smiles"].astype(str)
        self.motif_stats = motifs.set_index("motif_smiles")

        self._pair_map = {
            (row.key_a, row.key_b): row.prior for _, row in self.pair_counts.iterrows()
        }

        self.capacity = {m: len({t[0] for t in (ports or []) if isinstance(t, list) and len(t)==1})
                         for m, ports in self.attachments.items()}

    def pair_prior(self, ma: str, mb: str) -> float:
        if ma <= mb:
            return self._pair_map.get((ma, mb), 0.0)
        return self._pair_map.get((mb, ma), 0.0)

    def max_degree(self, m: str) -> int:
        return int(self.capacity.get(m, 4))
