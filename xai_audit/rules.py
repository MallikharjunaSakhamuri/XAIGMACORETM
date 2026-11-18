"""Dataset-specific medicinal-chemistry rule sets R_d.

These define the binary group label G(x), with G=0 compliant and G=1
non-compliant, used for the rule-partition parity analysis.

NOTE ON A MANUSCRIPT INCONSISTENCY
----------------------------------
The thresholds in Appendix Table A.3 and those stated in the Section 6.4 body
text disagree for two datasets:

    BACE   Table A.3: MW <= 450, LogP <= 3, TPSA 40-90
           Body text : MW 300-600, LogP 1-5
    ESOL   Table A.3: LogP <= 5, TPSA <= 140
           Body text : LogP <= 3, TPSA > 75

`CANONICAL_RULES` implements Appendix Table A.3, which is the formal
specification. `BODY_TEXT_RULES` implements the Section 6.4 wording so the
discrepancy can be quantified. One of the two must be removed from the
manuscript before resubmission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


def molecule_properties(mol: Chem.Mol) -> Dict[str, float]:
    """Physicochemical descriptors used by every rule set."""
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
    }


@dataclass(frozen=True)
class RuleSet:
    """A named compliance rule with an inclusive interval per property."""

    name: str
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]]
    extra: Optional[Callable[[Chem.Mol], bool]] = None

    def is_compliant(self, mol: Chem.Mol) -> bool:
        return self.is_compliant_from_properties(molecule_properties(mol), mol)

    def is_compliant_from_properties(
        self, props: Dict[str, float], mol: Chem.Mol
    ) -> bool:
        """Same decision as is_compliant, against descriptors already computed.

        Callers that evaluate several rule sets over the same molecule use this
        to avoid recomputing the descriptor block once per rule set. `mol` is
        still required because `extra` inspects the graph directly.
        """
        for key, (low, high) in self.bounds.items():
            value = props[key]
            if low is not None and value < low:
                return False
            if high is not None and value > high:
                return False
        if self.extra is not None and not self.extra(mol):
            return False
        return True

    def group_label(self, mol: Chem.Mol) -> int:
        """G(x): 0 for compliant, 1 for non-compliant."""
        return 0 if self.is_compliant(mol) else 1


def _qm9_elements(mol: Chem.Mol) -> bool:
    """QM9 canonical envelope: C, N, O, S only (hydrogens implicit)."""
    allowed = {1, 6, 7, 8, 16}
    return all(atom.GetAtomicNum() in allowed for atom in mol.GetAtoms())


CANONICAL_RULES: Dict[str, RuleSet] = {
    "clintox": RuleSet(
        name="Ro5_compliant",
        bounds={"MW": (None, 500), "LogP": (None, 5), "HBD": (None, 5), "HBA": (None, 10)},
    ),
    "bbbp": RuleSet(
        name="BBB_favorable",
        bounds={
            "MW": (None, 450), "LogP": (2, 5), "HBD": (None, 2),
            "HBA": (None, 6), "TPSA": (None, 90), "RotBonds": (None, 8),
        },
    ),
    "bace": RuleSet(
        name="BACE_favorable",
        bounds={
            "MW": (None, 450), "LogP": (None, 3), "HBD": (None, 3),
            "HBA": (None, 7), "TPSA": (40, 90), "RotBonds": (None, 8),
        },
    ),
    "esol": RuleSet(
        name="ESOL_favorable",
        bounds={
            "MW": (None, 500), "LogP": (None, 5), "HBD": (None, 5),
            "HBA": (None, 10), "TPSA": (None, 140), "RotBonds": (None, 10),
        },
    ),
    "qm9": RuleSet(
        name="QM9_envelope",
        bounds={"HeavyAtoms": (None, 9)},
        extra=_qm9_elements,
    ),
}


BODY_TEXT_RULES: Dict[str, RuleSet] = {
    "bace": RuleSet(
        name="BACE_inhibitor_window_body",
        bounds={"MW": (300, 600), "LogP": (1, 5)},
    ),
    "esol": RuleSet(
        name="ESOL_solubility_body",
        bounds={"LogP": (None, 3), "TPSA": (75, None)},
    ),
}
