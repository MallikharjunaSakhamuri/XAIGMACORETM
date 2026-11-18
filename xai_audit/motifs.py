"""Motif and functional-group definitions used throughout the audit.

The SMARTS patterns reproduce the functional groups named in Figure 5 and
Appendix A.1 of the manuscript. Group membership is binary per molecule:
a molecule belongs to a group if the pattern matches at least once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


# Pattern set. Keys are the labels used in the manuscript figures.
MOTIF_SMARTS: Dict[str, str] = {
    # Nitrogen containing (Appendix A.1a)
    "primary_amine": "[NX3;H2;!$(NC=[O,S,N]);!$(N[a]);!$(N[SX4](=O)=O);!$(N[PX4])]",
    "secondary_amine": "[NX3;H1;!$(NC=[O,S,N]);!$(N[a]);!$(N[SX4](=O)=O);!$(N[PX4])]",
    "tertiary_amine": "[NX3;H0;!$(NC=[O,S,N]);!$(N[a]);!$([N+]);!$(N[SX4](=O)=O);!$(N[PX4])]",
    "amine": "[NX3;!$(NC=[O,S,N]);!$(N[a]);!$(N[SX4](=O)=O);!$(N[PX4]);!$([NX3](=O)=O)]",
    "amide": "[NX3][CX3](=[OX1])",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    # Oxygen containing (Appendix A.1b)
    "alcohol": "[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]",
    "phenol": "[OX2H][c]",
    "ether": "[OD2]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])]",
    "ester": "[CX3](=[OX1])[OX2][#6;!$(C=[O,N,S])]",
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxyl": "[CX3](=O)[OX2H1,OX1-]",
    # Halogen containing (Appendix A.1c)
    "fluoro": "[F]",
    "chloro": "[Cl]",
    "bromo": "[Br]",
    "iodo": "[I]",
    "halogen": "[F,Cl,Br,I]",
    # Sulfur containing (Appendix A.1d)
    "thiol": "[SX2H]",
    "sulfide": "[#16X2H0][#6]",
    "sulfone": "[$([SX4](=[OX1])(=[OX1])([#6])[#6])]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "sulfate": "[$([SX4](=[OX1])(=[OX1])([OX2])[OX2])]",
    # Ring and scaffold level motifs referenced in Figures 2 and 3
    "aromatic_ring": "[a]1[a][a][a][a][a]1",
    "heteroaromatic": "[a;!c]",
    "fused_ring": "[R2]",
}


# Groups reported per dataset in the manuscript figures. Keeping this explicit
# means Table A and Table B cover exactly the groups already discussed, rather
# than the whole library.
DATASET_GROUPS: Dict[str, List[str]] = {
    "bace": [
        "fluoro", "sulfonamide", "chloro", "bromo",
        "alcohol", "ether", "amide", "aromatic_ring",
    ],
    "esol": [
        "primary_amine", "aldehyde", "chloro", "ketone",
        "alcohol", "ether", "halogen", "aromatic_ring",
    ],
    "qm9": [
        "alcohol", "phenol", "fluoro", "ether",
        "amine", "carbonyl",
    ],
    "clintox": [
        "carboxyl", "amine", "amide", "halogen",
        "alcohol", "aromatic_ring",
    ],
    "bbbp": [
        "amine", "amide", "halogen", "alcohol",
        "ether", "aromatic_ring",
    ],
}


@dataclass(frozen=True)
class MotifLibrary:
    """Compiled SMARTS patterns with match helpers."""

    patterns: Dict[str, Chem.Mol]

    @classmethod
    def build(cls, names: Optional[Sequence[str]] = None) -> "MotifLibrary":
        selected = list(names) if names is not None else list(MOTIF_SMARTS)
        compiled = {}
        for name in selected:
            if name not in MOTIF_SMARTS:
                raise KeyError(f"unknown motif '{name}'")
            pattern = Chem.MolFromSmarts(MOTIF_SMARTS[name])
            if pattern is None:
                raise ValueError(f"SMARTS failed to compile for '{name}'")
            compiled[name] = pattern
        return cls(patterns=compiled)

    @property
    def names(self) -> List[str]:
        return list(self.patterns)

    def membership(self, mol: Chem.Mol) -> Dict[str, int]:
        """Binary membership indicator M_x^m for every motif in the library."""
        if mol is None:
            return {name: 0 for name in self.patterns}
        return {
            name: int(mol.HasSubstructMatch(pattern))
            for name, pattern in self.patterns.items()
        }

    def matched_atoms(self, mol: Chem.Mol, name: str) -> List[Tuple[int, ...]]:
        """Atom index tuples V_m(x) for each occurrence of a motif."""
        if mol is None:
            return []
        return list(mol.GetSubstructMatches(self.patterns[name]))

    def atom_set(self, mol: Chem.Mol, name: str) -> List[int]:
        """Flat, de-duplicated atom index set for a motif within one molecule."""
        atoms: set = set()
        for match in self.matched_atoms(mol, name):
            atoms.update(match)
        return sorted(atoms)


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse and sanitize a SMILES string, returning None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    return mol
