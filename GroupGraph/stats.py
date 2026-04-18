
import os, json, argparse
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
        if gi < gj: inter.append((i, j, gi, gj, b.GetIdx()))
        else:       inter.append((j, i, gj, gi, b.GetIdx()))
    # unique
    seen = set(); out = []
    for i,j,gi,gj,bi in inter:
        key = (i,j,gi,gj)
        if key in seen: continue
        seen.add(key)
        out.append((i,j,gi,gj,bi))
    return out

def bond_order_label(bond: rdchem.Bond) -> str:
    bt = bond.GetBondType()
    if bt == rdchem.BondType.SINGLE:  return "single"
    if bt == rdchem.BondType.DOUBLE:  return "double"
    if bt == rdchem.BondType.TRIPLE:  return "triple"
    if bt == rdchem.BondType.AROMATIC: return "aromatic"
    # fallback (rare)
    if bond.GetIsAromatic(): return "aromatic"
    return "single"

# ---------------------------- main analysis ----------------------------
def analyze(in_csv, smiles_col, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {in_csv}")
    if limit is not None:
        df = df.head(int(limit)).copy()

    # Counters
    pair_counts = defaultdict(lambda: Counter(single=0, double=0, triple=0, aromatic=0))
    pair_by_anchor = defaultdict(lambda: Counter(single=0, double=0, triple=0, aromatic=0))

    # Iterate
    for smi in tqdm(df[smiles_col].astype(str).tolist(), desc="Analyzing"):
        mol = get_mol(smi)
        if mol is None: continue

        groups = partition_into_motifs(mol)
        if not groups: continue

        # Build submol per group so motif SMILES are stable + local ports known
        group_info = []
        for g in groups:
            sub, orig2local, local2orig = submol_with_local_mapping(mol, g)
            msmi = Chem.MolToSmiles(sub, kekuleSmiles=True, isomericSmiles=False)
            group_info.append((tuple(g), msmi, sub, orig2local, local2orig))

        inter = inter_motif_bonds(mol, [g for g, *_ in group_info])
        if not inter: continue

        # Map original atom -> (motif_id, local_idx)
        atom2g_local = {}
        for gi, (g_atoms, _smi, _sub, orig2local, _l2o) in enumerate(group_info):
            for a in g_atoms:
                atom2g_local[a] = (gi, orig2local[a])

        for ai, aj, gi, gj, bond_idx in inter:
            a_motif = group_info[gi][1]
            b_motif = group_info[gj][1]
            if a_motif <= b_motif:
                ma, mb = a_motif, b_motif
                li, lj = atom2g_local[ai][1], atom2g_local[aj][1]
                ai_elem = group_info[gi][2].GetAtomWithIdx(li).GetSymbol()
                aj_elem = group_info[gj][2].GetAtomWithIdx(lj).GetSymbol()
            else:
                ma, mb = b_motif, a_motif
                lj, li = atom2g_local[ai][1], atom2g_local[aj][1]
                aj_elem = group_info[gi][2].GetAtomWithIdx(lj).GetSymbol()
                ai_elem = group_info[gj][2].GetAtomWithIdx(li).GetSymbol()

            bond = mol.GetBondWithIdx(bond_idx)
            label = bond_order_label(bond)  # single/double/triple/aromatic

            pair_counts[(ma, mb)][label] += 1
            pair_by_anchor[(ma, mb, ai_elem, aj_elem)][label] += 1

    # Dump CSVs
    rows = []
    for (ma, mb), c in pair_counts.items():
        total = sum(c.values())
        rows.append({
            "motif_a": ma, "motif_b": mb,
            "single": c["single"], "double": c["double"],
            "triple": c["triple"], "aromatic": c["aromatic"],
            "total": total
        })
    pd.DataFrame(rows).sort_values("total", ascending=False).to_csv(
        os.path.join(out_dir, "edge_orders.csv"), index=False
    )

    rows2 = []
    for (ma, mb, ea, eb), c in pair_by_anchor.items():
        total = sum(c.values())
        rows2.append({
            "motif_a": ma, "motif_b": mb,
            "elem_a": ea, "elem_b": eb,
            "single": c["single"], "double": c["double"],
            "triple": c["triple"], "aromatic": c["aromatic"],
            "total": total
        })
    pd.DataFrame(rows2).sort_values("total", ascending=False).to_csv(
        os.path.join(out_dir, "edge_orders_by_anchor.csv"), index=False
    )

    # Quick summary
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("Top motif pairs by total inter-motif bonds:\n")
        top = sorted(rows, key=lambda r: r["total"], reverse=True)[:50]
        for r in top:
            tot = r["total"]
            if tot == 0: continue
            f.write(
                f"- {r['motif_a']}  —  {r['motif_b']}: "
                f"single {r['single']}, double {r['double']}, "
                f"triple {r['triple']}, aromatic {r['aromatic']} (total {tot})\n"
            )
        f.write("\nRule of thumb: inter-motif bonds are almost always SINGLE; "
                "use these counts to whitelist rare DOUBLE cases (and ignore aromatic across motifs).\n")

def main():
    ap = argparse.ArgumentParser(description="Analyze inter-motif bond orders from a SMILES CSV.")
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--smiles_col", type=str, default="smiles")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    analyze(args.in_csv, args.smiles_col, args.out_dir, limit=args.limit)

if __name__ == "__main__":
    main()

