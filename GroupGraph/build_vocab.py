
import os
import json
import argparse
from collections import defaultdict, Counter
from functools import lru_cache

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog('rdApp.*')

# ---------------------------- SMARTS split rules (same as GroupGraph) ----------------------------
SPLIT_SMARTS = [
    '[#6;X3&+]', '[#6;X3&-]','[#8;X2]', '[#8;-]', '[#8;+]',
    '[#16;X2]', '[#16;X3&+]', '[#16;X1&-]',
    '[#7;X3&+0]', '[#7;X4&+]', '[#7;X5&2+]',  '[#7;-]', '[#15;X4&+]',
    '[As;X5]', '[As;X3]','[As;X4&+]',
    '[Fe]', '[Al]','[Sn]', '[#0]', '[Mg]', '[Si]', '[Se]',
    '[F]', '[Cl]', '[Br]', '[I]', '[#5]', '[Ge]', '[Ni]','[Se]',
    '[Ca]', '[Cu]','[Li]', '[Ru]', '[Co]', '[Pt]','[Ir]', '[Pd]', '[W]',
    '[#6]=[#6]', '[#6]#[#6]', '[#6]=[#7]', '[#6]=[#16]','[#6]#[#7]', '[#6]=[#15]',
    '[#7;X3&+][#8;X1&+0]', '[#7]=[#7]', '[#7]#[#7]', '[#7]=[#8]','[#6]=[#8]',
    '[#16]=[#8]','[#16]=[#16]','[#16]=[#7]', '[#15]=[#8]','[As]=[#16]','[As]=[#8]',
    '[#7](=[#8])(=[#8])', '[#6]=[#7;+]=[#7;-]', '[#7]=[#7;+]=[#7;-]',
    '[#16](=[#8])(=[#8])',  '[#15](=[#8])(=[#8])'
]
# ---------------------------- SMARTS split rules (same as GroupGraph) ----------------------------

SPLIT_PATTS = [Chem.MolFromSmarts(s) for s in SPLIT_SMARTS]
def get_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    return mol

def fused_aromatic_components(mol):
    ri = mol.GetRingInfo()
    if not ri: return []
    atom_rings = list(ri.AtomRings())
    bond_rings = list(ri.BondRings())
    aro = []
    for ar, br in zip(atom_rings, bond_rings):
        if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in br):
            aro.append(set(ar))
    # merge fused aromatic rings into components
    merged = True
    comps = aro
    while merged:
        merged = False
        out = []
        while comps:
            a = comps.pop()
            hit = None
            for i, b in enumerate(comps):
                if a & b:
                    hit = i; break
            if hit is None:
                out.append(a)
            else:
                comps[hit] |= a
                merged = True
        comps = out
    return [tuple(sorted(list(s))) for s in comps]

def collect_split_hits_non_aromatic(mol, aro_atoms):
    out = defaultdict(list)
    for patt in SPLIT_PATTS:
        if patt is None: continue
        try:
            matches = mol.GetSubstructMatches(patt)
        except Exception:
            continue
        for m in matches:
            if set(m).intersection(aro_atoms):
                continue
            out[Chem.MolToSmarts(patt)].append(tuple(sorted(m)))

    return dict(out)

def dedup_non_overlapping(groups):
    groups = sorted(set(groups), key=lambda t: (-len(t), t))
    assigned = set(); res = []
    for g in groups:
        if assigned & set(g): continue
        res.append(g); assigned |= set(g)
    return res

def bfs_components_on_subset(mol, allowed):
    visited = set(); comps = []
    for a in allowed:
        if a in visited: continue
        comp, stack = [], [a]
        visited.add(a)
        while stack:
            u = stack.pop(); comp.append(u)
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                v = nb.GetIdx()
                if v in allowed and v not in visited:
                    visited.add(v); stack.append(v)
        comps.append(tuple(sorted(comp)))
    return comps

def partition_into_motifs(mol):
    aro = fused_aromatic_components(mol)
    aro_atoms = set(a for comp in aro for a in comp)
    split_hits = collect_split_hits_non_aromatic(mol, aro_atoms)
    seeds = list(aro)
    for lst in split_hits.values():
        seeds += list(lst)
    seeds = dedup_non_overlapping(seeds)
    assigned = set(a for g in seeds for a in g)
    leftover = set(range(mol.GetNumAtoms())) - assigned
    fatty = bfs_components_on_subset(mol, leftover) if leftover else []
    return list(seeds) + list(fatty)

def submol_with_local_mapping(mol, atom_tuple):
    atom_tuple = tuple(sorted(atom_tuple))
    aset = set(atom_tuple)
    em = Chem.EditableMol(Chem.Mol())
    orig2local = {}
    for aidx in atom_tuple:
        a = mol.GetAtomWithIdx(aidx)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetChiralTag(a.GetChiralTag())
        na.SetIsAromatic(a.GetIsAromatic())
        local_idx = em.AddAtom(na)
        orig2local[aidx] = local_idx
    for i in atom_tuple:
        ai = mol.GetAtomWithIdx(i)
        for nb in ai.GetNeighbors():
            j = nb.GetIdx()
            if j in aset and i < j:
                b = mol.GetBondBetweenAtoms(i, j)
                em.AddBond(orig2local[i], orig2local[j], b.GetBondType())
    sub = em.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        pass
    local2orig = list(atom_tuple)
    return sub, orig2local, local2orig

def inter_motif_bonds(mol, groups):
    atom2group = {}
    for gi, g in enumerate(groups):
        for a in g: atom2group[a] = gi
    inter = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        gi, gj = atom2group.get(i), atom2group.get(j)
        if gi is None or gj is None or gi == gj: continue
        if gi < gj: inter.append((i, j, gi, gj))
        else:       inter.append((j, i, gj, gi))
    return sorted(set(inter))

def bond_order_num(bond):
    bt = bond.GetBondType()
    if bt == rdchem.BondType.SINGLE: return 1.0
    if bt == rdchem.BondType.DOUBLE: return 2.0
    if bt == rdchem.BondType.TRIPLE: return 3.0
    if bt == rdchem.BondType.AROMATIC: return 1.5
    return 1.0

@lru_cache(maxsize=None)
def canonical_ref_for_smiles(smi: str):
    """Return (mol_ref_canon, perm) with canonical atom order for that motif SMILES."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:
        pass
    ranks = Chem.CanonicalRankAtoms(m, includeChirality=False)
    perm = np.argsort(ranks)
    m_canon = Chem.RenumberAtoms(m, perm.tolist())
    return m_canon, perm

def occ_local_to_canonical_indices(smi: str, occ_submol: Chem.Mol, occ_local_idxs: list[int]) -> list[int]:
    """
    Map local indices from an occurrence submol to canonical indices of motif smi.
    Returns canonical indices; falls back to local if no complete match is found.
    """
    mol_ref, _ = canonical_ref_for_smiles(smi)
    if mol_ref is None:
        return occ_local_idxs  # fallback
    # match: tuple of submol indices in the order of mol_ref atoms
    matches = occ_submol.GetSubstructMatches(mol_ref)
    if not matches:
        return occ_local_idxs  # fallback
    # choose the first match; for highly symmetric motifs you could score matches by invariants
    match = matches[0]
    if len(match) != mol_ref.GetNumAtoms():
        return occ_local_idxs
    inv = {sub_idx: canon_idx for canon_idx, sub_idx in enumerate(match)}
    try:
        return [inv[i] for i in occ_local_idxs]
    except KeyError:
        return occ_local_idxs  # partial fallback

# ---------------------------- Extraction (attachments only, canonicalized) ----------------------------
def extract_canonical_attachments_from_mol(mol):
    """
    Returns: dict motif_smiles -> list of canonical attachment atom index lists (per edge)
    (one-element lists for single-atom ports in this implementation).
    """
    out = defaultdict(list)

    groups = partition_into_motifs(mol)
    if not groups:
        return out

    # Build submols and keep local mapping per group
    group_info = []
    for g in groups:
        sub, orig2local, local2orig = submol_with_local_mapping(mol, g)
        smi = Chem.MolToSmiles(sub, kekuleSmiles=True, isomericSmiles=False)
        group_info.append((tuple(g), smi, sub, orig2local, local2orig))

    # Map each inter-motif bond to local attachment atoms
    atom2g_local = {}
    for gi, (g_atoms, _smi, _sub, orig2local, _l2o) in enumerate(group_info):
        for a in g_atoms:
            atom2g_local[a] = (gi, orig2local[a])

    inter = inter_motif_bonds(mol, [g for g, *_ in group_info])
    for (ai, aj, gi, gj) in inter:
        smi_i = group_info[gi][1]
        smi_j = group_info[gj][1]

        gi_local = atom2g_local[ai][1]
        gj_local = atom2g_local[aj][1]

        # Remap local occurrence indices -> canonical indices
        canon_i = occ_local_to_canonical_indices(smi_i, group_info[gi][2], [gi_local])[0]
        canon_j = occ_local_to_canonical_indices(smi_j, group_info[gj][2], [gj_local])[0]

        out[smi_i].append([canon_i])
        out[smi_j].append([canon_j])

    return out

# ---------------------------- CLI / main ----------------------------
def read_subset_smiles(path):
    if path is None: return None
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def canon_smiles(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None

def main():
    ap = argparse.ArgumentParser(description="Build ONLY attachments.json with canonicalized port indices.")
    ap.add_argument("--in_csv", type=str, required=True, help="Input CSV with a SMILES column")
    ap.add_argument("--smiles_col", type=str, default="smiles", help="SMILES column name")
    ap.add_argument("--out_json", type=str, required=True, help="Path to write attachments.json")
    ap.add_argument("--subset_smiles", type=str, default=None,
                    help="Optional text file (one SMILES/line) to restrict the dataset (training split).")
    ap.add_argument("--canonicalize_filter", action="store_true",
                    help="Canonicalize both CSV SMILES and subset file before filtering.")
    ap.add_argument("--min_motif_count", type=int, default=1,
                    help="Minimum number of molecules a motif must appear in to be kept")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional limit on rows for speed")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in {args.in_csv}")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    # Optional canonicalization before filtering
    if args.canonicalize_filter:
        tqdm.pandas(desc="Canonicalizing CSV SMILES")
        df["_canon"] = df[args.smiles_col].astype(str).progress_map(canon_smiles)

    # Optional subset filtering
    keep_list = read_subset_smiles(args.subset_smiles)
    if keep_list is not None:
        if args.canonicalize_filter:
            keep_canon = set(filter(None, (canon_smiles(s) for s in keep_list)))
            df = df[df["_canon"].isin(keep_canon)].copy()
        else:
            keep_set = set(keep_list)
            df = df[df[args.smiles_col].astype(str).isin(keep_set)].copy()
        if len(df) == 0:
            raise RuntimeError(
                "Training filter kept 0 rows.\n"
                "• Ensure the subset SMILES came from the SAME CSV & column.\n"
                "• Or pass --canonicalize_filter to normalize SMILES before matching."
            )

    # Iterate and collect canonicalized attachment atoms
    attachments_raw = defaultdict(list)  # motif_smiles -> list of [canon_atom] (duplicates allowed)
    smiles_iter = (df["_canon"] if args.canonicalize_filter else df[args.smiles_col]).astype(str).tolist()

    for smi in tqdm(smiles_iter, desc="Scanning molecules"):
        mol = get_mol(smi)
        if mol is None:
            continue
        att = extract_canonical_attachments_from_mol(mol)
        for m, lst in att.items():
            attachments_raw[m].extend(lst)
    motif_coverage = Counter()

    for smi in tqdm(smiles_iter, desc="Estimating motif coverage"):
        mol = get_mol(smi)
        if mol is None:
            continue
        groups = partition_into_motifs(mol)
        seen = set()
        for g in groups:
            sub, _, _ = submol_with_local_mapping(mol, g)
            ms = Chem.MolToSmiles(sub, kekuleSmiles=True, isomericSmiles=False)
            seen.add(ms)
        for ms in seen:
            motif_coverage[ms] += 1

    # Filter by min_motif_count, then deduplicate lists
    VA = {}
    for m, lst in attachments_raw.items():
        if motif_coverage[m] >= args.min_motif_count:
            uniq = sorted(set(tuple(x) for x in lst))  # each x is a one-atom list [i] here
            VA[m] = [list(t) for t in uniq]

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(VA, f, indent=2)

    print(f"Saved canonicalized attachments to {args.out_json}")
    print(f"Motifs kept: {len(VA)} (min_motif_count={args.min_motif_count})")

if __name__ == "__main__":
    main() 
