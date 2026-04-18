import os
import argparse
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog('rdApp.*')

# ---------------------------- SMARTS split rules (same as GG) ----------------------------
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
    out = []
    for patt in SPLIT_PATTS:
        if patt is None: continue
        try:
            matches = mol.GetSubstructMatches(patt)
        except Exception:
            continue
        for m in matches:
            if set(m).intersection(aro_atoms):
                continue
            out.append(tuple(sorted(m)))
    return out

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
    seeds = list(aro) + list(split_hits)
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
    # unique
    seen = set(); out = []
    for i,j,gi,gj in inter:
        key = (i,j,gi,gj)
        if key in seen: continue
        seen.add(key)
        out.append((i,j,gi,gj))
    return out

def analyze(in_csv, smiles_col, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' was not found in {in_csv}")
    if limit is not None:
        df = df.head(int(limit)).copy()

    motif_per_mol_counts = defaultdict(list)
    pair_per_mol_counts = defaultdict(list)
    per_mol_rows = []

    for idx, smi in tqdm(list(enumerate(df[smiles_col].astype(str).tolist())), desc="Processing"):
        mol = get_mol(smi)
        if mol is None:
            per_mol_rows.append({
                "mol_index": idx, "smiles": smi,
                "top_motif": None, "top_motif_count": 0,
                "top_pair_a": None, "top_pair_b": None, "top_pair_count": 0
            })
            continue

        groups = partition_into_motifs(mol)
        if not groups:
            per_mol_rows.append({
                "mol_index": idx, "smiles": smi,
                "top_motif": None, "top_motif_count": 0,
                "top_pair_a": None, "top_pair_b": None, "top_pair_count": 0
            })
            continue

        group_info = []
        for g in groups:
            sub, _, _ = submol_with_local_mapping(mol, g)
            msmi = Chem.MolToSmiles(sub, kekuleSmiles=True, isomericSmiles=False)
            group_info.append((tuple(g), msmi))

        motif_counts_this_mol = Counter([msmi for _, msmi in group_info])

        for msmi, c in motif_counts_this_mol.items():
            motif_per_mol_counts[msmi].append(c)
        inter = inter_motif_bonds(mol, [g for g, _ in group_info])
        pair_counts_this_mol = Counter()
        if inter:
            atom2group = {}
            for gi, (g_atoms, _msmi) in enumerate(group_info):
                for a in g_atoms:
                    atom2group[a] = gi
            for (ai, aj, gi, gj) in inter:
                ma = group_info[gi][1]
                mb = group_info[gj][1]
                if ma <= mb:
                    key = (ma, mb)
                else:
                    key = (mb, ma)
                pair_counts_this_mol[key] += 1

        for key, c in pair_counts_this_mol.items():
            pair_per_mol_counts[key].append(c)

        if motif_counts_this_mol:
            top_motif, top_motif_count = max(motif_counts_this_mol.items(), key=lambda kv: kv[1])
        else:
            top_motif, top_motif_count = (None, 0)
        if pair_counts_this_mol:
            (pa, pb), top_pair_count = max(pair_counts_this_mol.items(), key=lambda kv: kv[1])
        else:
            (pa, pb), top_pair_count = (None, None), 0

        per_mol_rows.append({
            "mol_index": idx,
            "smiles": smi,
            "top_motif": top_motif,
            "top_motif_count": top_motif_count,
            "top_pair_a": pa,
            "top_pair_b": pb,
            "top_pair_count": top_pair_count
        })


    motif_rows = []
    for msmi, counts in motif_per_mol_counts.items():
        s = pd.Series(counts, dtype=float)
        motif_rows.append({
            "motif_smiles": msmi,
            "mean_per_mol": float(s.mean()),
            "std_per_mol": float(s.std(ddof=0)),
            "max_in_any_mol": int(s.max()),
            "n_molecules_with": int(s.size),
            "total_occurrences": int(s.sum())
        })
    motifs_df = pd.DataFrame(motif_rows).sort_values(
        ["mean_per_mol", "n_molecules_with"], ascending=[False, False]
    )


    pair_rows = []
    for (ma, mb), counts in pair_per_mol_counts.items():
        s = pd.Series(counts, dtype=float)
        pair_rows.append({
            "motif_a": ma,
            "motif_b": mb,
            "mean_per_mol": float(s.mean()),
            "std_per_mol": float(s.std(ddof=0)),
            "max_in_any_mol": int(s.max()),
            "n_molecules_with": int(s.size),
            "total_edges": int(s.sum())
        })
    pairs_df = pd.DataFrame(pair_rows).sort_values(
        ["mean_per_mol", "n_molecules_with"], ascending=[False, False]
    )

    per_mol_df = pd.DataFrame(per_mol_rows)
    motifs_path = os.path.join(out_dir, "motifs_redundancy.csv")
    pairs_path = os.path.join(out_dir, "pairs_redundancy.csv")
    per_mol_path = os.path.join(out_dir, "per_molecule_max.csv")

    motifs_df.to_csv(motifs_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)
    per_mol_df.to_csv(per_mol_path, index=False)

    print("Saved:")
    print(" ", motifs_path)
    print(" ", pairs_path)
    print(" ", per_mol_path)
    print("\nColumns explained:")
    print("  motifs_redundancy.csv:")
    print("    - mean_per_mol: average count of this motif among molecules where it appears")
    print("    - std_per_mol:   std dev of that count")
    print("    - max_in_any_mol: max repeats of this motif inside a single molecule")
    print("    - n_molecules_with: number of molecules where this motif was seen")
    print("    - total_occurrences: total motif count across the dataset")
    print("  pairs_redundancy.csv (same semantics but for inter-motif edges).")
    print("  per_molecule_max.csv: for each molecule, the most repeated motif and pair.")

def main():
    ap = argparse.ArgumentParser(description="Redundancy stats for motifs and motif pairs per molecule.")
    ap.add_argument("--in_csv", required=True, type=str)
    ap.add_argument("--smiles_col", default="smiles", type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--limit", default=None, type=int, help="Optionally cap the number of rows for speed.")
    args = ap.parse_args()

    analyze(args.in_csv, args.smiles_col, args.out_dir, limit=args.limit)

if __name__ == "__main__":
    main()
