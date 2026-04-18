import random
from typing import Iterable, Optional

import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import one_hot

from rdkit import Chem
from rdkit.Chem import Crippen

BOND2IDX = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}

class SMILESStream(IterableDataset):

    def __init__(
        self,
        path: str,
        type_list,
        smiles_col: Optional[str] = None,
        shuffle_buffer: int = 0,
        target: str = "logp",
        max_atoms: Optional[int] = None,
        seed: int = 1337,
    ):
        super().__init__()
        self.path = path
        self.type_list = list(type_list)
        self.type_map = {s: i for i, s in enumerate(self.type_list)}
        self.smiles_col = smiles_col
        self.shuffle_buffer = shuffle_buffer
        self.target = target.lower()
        self.max_atoms = max_atoms
        self.seed = seed

    def _iter_smiles_lines(self) -> Iterable[str]:
        if self.path.lower().endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(self.path)
            col = self.smiles_col or ("smiles" if "smiles" in df.columns else df.columns[0])
            for s in df[col].astype(str):
                yield s
        else:
            with open(self.path, "r") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        yield ln

    def _shuffle_stream(self, it: Iterable[str]) -> Iterable[str]:
        if self.shuffle_buffer <= 0:
            yield from it
            return
        buf, rng = [], random.Random(self.seed)
        for s in it:
            buf.append(s)
            if len(buf) >= self.shuffle_buffer:
                rng.shuffle(buf)
                while buf:
                    yield buf.pop()
        rng.shuffle(buf)
        while buf:
            yield buf.pop()

    def __iter__(self):
        it = self._iter_smiles_lines()
        it = self._shuffle_stream(it) if self.shuffle_buffer > 0 else it

        for i, smi in enumerate(it):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                Chem.SanitizeMol(m)
            except Exception:
                continue

            if self.max_atoms and m.GetNumAtoms() > self.max_atoms:
                continue

            # Atom type one-hot (skip molecules with out-of-vocab atom symbols)
            idxs = []
            bad = False
            for a in m.GetAtoms():
                sym = a.GetSymbol()
                if sym not in self.type_map:
                    bad = True
                    break
                idxs.append(self.type_map[sym])
            if bad:
                continue

            x = one_hot(torch.tensor(idxs), num_classes=len(self.type_list)).to(torch.float)

            # Undirected edges and bond types
            rows, cols, bond_types = [], [], []
            for b in m.GetBonds():
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                rows += [u, v]
                cols += [v, u]
                bt = BOND2IDX.get(b.GetBondType(), 0)
                bond_types += [bt, bt]

            if rows:
                edge_index = torch.tensor([rows, cols], dtype=torch.long)
                # Sort edges for determinism
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = one_hot(torch.tensor(bond_types)[perm], num_classes=4).to(torch.float)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 4), dtype=torch.float)

            # Label y
            if self.target == "logp":
                # Use AddHs to match the standard Crippen calculation convention
                y_val = float(Crippen.MolLogP(Chem.AddHs(m)))
            else:
                # Extendable for future targets
                y_val = 0.0
            y = torch.tensor([[y_val]], dtype=torch.float)

            yield Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                name=smi,
                idx=i,
            )


def make_stream_loader(path: str, cfg: dict, shuffle: bool) -> DataLoader:
    ds = SMILESStream(
        path=path,
        type_list=cfg["type_list"],
        smiles_col=cfg.get("smiles_col"),
        shuffle_buffer=(cfg.get("shuffle_buffer", 0) if shuffle else 0),
        target=cfg.get("target", "logp"),
        max_atoms=cfg.get("max_atoms"),
        seed=cfg.get("seed", 1337),
    )
    return DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 0),
        persistent_workers=cfg.get("num_workers", 0) > 0,
        pin_memory=True,
    )
