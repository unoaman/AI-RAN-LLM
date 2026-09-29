"""Cell identity map: RAN identities <-> model cell indices (0..63).

The model only knows cell *indices*. A real RAN identifies cells by PCI
(physical cell id, 0..1007, reused across the network), NR Cell Identity (NCI,
36 bit, unique per PLMN), NR-CGI (PLMN + NCI), and belongs to a gNB / E2 node.
The cell map is a small JSON file listing every cell the xApp may see::

    {
      "plmn": "00101",
      "cells": [
        {"index": 0, "pci": 1, "nci": "0x66C000", "gnb": "gnb-411",
         "e2_node_id": "gnbd_001_001_00019b_0", "neighbours": [1]},
        {"index": 1, "pci": 2, "nci": "0x66C001", "gnb": "gnb-411",
         "e2_node_id": "gnbd_001_001_00019b_0", "neighbours": [0]}
      ]
    }

* ``index`` (0..63) is the model's cell id; omit it to number cells in order.
* ``neighbours`` (optional) is the neighbour relation table: when present, the
  controller only commands handovers to listed neighbours.
* ``plmn`` can be set globally or per cell.

PCIs must be unique among the cells in one map (a UE report only carries the
PCI of neighbours). If they are not, add the frequency to your adapter and
resolve by NCI instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..tokenizer import MAX_CELLS
from .messages import CellMeas, CellRef


@dataclass
class CellMap:
    cells: dict[int, CellRef] = field(default_factory=dict)
    neighbours: dict[int, set[int]] = field(default_factory=dict)
    auto_add: bool = False            # assign indices to unknown PCIs on the fly (lab use)

    def __post_init__(self):
        self._by_pci = {c.pci: c for c in self.cells.values() if c.pci is not None}
        self._by_nci = {c.nci: c for c in self.cells.values() if c.nci is not None}
        if len(self._by_pci) != sum(c.pci is not None for c in self.cells.values()):
            raise ValueError("cell map: PCIs must be unique")

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict, auto_add: bool = False) -> "CellMap":
        plmn = d.get("plmn")
        cells, neigh = {}, {}
        for i, c in enumerate(d.get("cells", [])):
            idx = int(c.get("index", i))
            if not 0 <= idx < MAX_CELLS:
                raise ValueError(f"cell index {idx} outside 0..{MAX_CELLS - 1}")
            if idx in cells:
                raise ValueError(f"duplicate cell index {idx}")
            nci = c.get("nci")
            cells[idx] = CellRef(index=idx, pci=None if c.get("pci") is None else int(c["pci"]),
                                 nci=None if nci is None else (int(nci, 0) if isinstance(nci, str) else int(nci)),
                                 plmn=c.get("plmn", plmn), gnb=c.get("gnb"), e2_node_id=c.get("e2_node_id"))
            if "neighbours" in c:
                neigh[idx] = {int(n) for n in c["neighbours"]}
        return cls(cells, neigh, auto_add)

    @classmethod
    def load(cls, path: str, auto_add: bool = False) -> "CellMap":
        with open(path) as f:
            return cls.from_dict(json.load(f), auto_add)

    def to_dict(self) -> dict:
        out = []
        for idx in sorted(self.cells):
            c = self.cells[idx].to_dict()
            if c.get("nci") is not None:
                c["nci"] = hex(c["nci"])
            if idx in self.neighbours:
                c["neighbours"] = sorted(self.neighbours[idx])
            out.append(c)
        return {"cells": out}

    # ---- lookups -----------------------------------------------------------
    def resolve(self, cell: CellMeas) -> CellRef | None:
        """Report cell -> CellRef (NCI preferred, then PCI). Unknown -> None,
        or a new index when `auto_add` is on and a PCI is given."""
        if cell.nci is not None and cell.nci in self._by_nci:
            return self._by_nci[cell.nci]
        if cell.pci is not None and cell.pci in self._by_pci:
            return self._by_pci[cell.pci]
        if self.auto_add and cell.pci is not None:
            free = [i for i in range(MAX_CELLS) if i not in self.cells]
            if not free:
                return None
            ref = CellRef(index=free[0], pci=cell.pci, nci=cell.nci)
            self.cells[ref.index] = ref
            self._by_pci[ref.pci] = ref
            if ref.nci is not None:
                self._by_nci[ref.nci] = ref
            return ref
        return None

    def by_pci(self, pci: int) -> CellRef | None:
        return self._by_pci.get(pci)

    def __getitem__(self, index: int) -> CellRef:
        return self.cells[index]

    def is_neighbour(self, serving: int, target: int) -> bool:
        """True if no neighbour list is configured for `serving`, or target is in it."""
        return serving not in self.neighbours or target in self.neighbours[serving]

    @classmethod
    def for_simulator(cls, n_cells: int, plmn: str = "00101", gnb: str = "sim-gnb") -> "CellMap":
        """Identity map used by the fake gNB: index i <-> PCI i+1, NCI 0x1000+i."""
        return cls({i: CellRef(index=i, pci=i + 1, nci=0x1000 + i, plmn=plmn, gnb=gnb) for i in range(n_cells)})
