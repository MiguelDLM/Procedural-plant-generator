"""
Perturbation check for core.relevance: every field a rule marks as not applicable is changed drastically,
and the generated geometry must stay bit-identical. Used by the test suite (subset) and runnable in full:

    python -m tests.relevance_check
"""

import copy
import dataclasses
import os
import sys
from enum import Enum

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import presets as P                     # noqa: E402
from core.relevance import inactive_fields         # noqa: E402

# Representative presets: each exercises a different set of inactive rules
CASES = {
    "Cactus": ["opuntia_ficus_indica", "mammillaria_hahniana", "echinocactus_grusonii", "lophophora_williamsii"],
    "Rosette": ["agave_americana", "echeveria_elegans"],
    "Flower": ["helianthus_annuus", "rosa_canina", "magnolia_grandiflora", "tulipa_gesneriana"],
    "Tree": ["phoenix_canariensis", "pinus_sylvestris", "quercus_robur"],
    "Vine": ["ipomoea_purpurea", "hedera_helix", "vitis_vinifera", "cucurbita_pepo"],
    "Fruit": ["malus_domestica", "vitis_vinifera", "solanum_lycopersicum", "citrullus_lanatus"],
    "Vegetable": ["daucus_carota", "solanum_tuberosum", "brassica_oleracea_italica"],
}
MATERIAL_ONLY = ("_color", "equator_", "scars", "blush_tip", "guide_contrast", "stem_scars")


def _signature(form: str, preset) -> list:
    """Arrays that fully describe the generated geometry."""
    if form == "Cactus":
        from core.cactus import CactusEngine
        r = CactusEngine(preset.profile).generate(seed=3, detail=0.3, spine_budget=300, with_roots=True)
        return [r.stem.vertices, r.spines.vertices, r.roots.vertices if r.roots is not None else np.zeros(0)]
    if form == "Rosette":
        from core.rosette import RosetteEngine
        r = RosetteEngine(preset.profile).generate(seed=3, detail=0.3, with_roots=True)
        return [r.leaves.vertices, r.armature.vertices, r.stem.vertices,
                r.roots.vertices if r.roots is not None else np.zeros(0)]
    if form == "Vegetable":
        from core.vegetable import VegetableEngine
        r = VegetableEngine(preset.profile, preset.leaf, preset.venation).generate(seed=3, detail=0.5)
        return [r.root.vertices, r.stems.vertices, r.head.vertices, r.leaves.vertices]
    if form == "Fruit":
        from core.fruit import hanging_fruit
        return [hanging_fruit(preset.fruit, 0.5, 3).vertices]
    if form == "Vine":
        from core.vine import VineEngine, guide_shape
        from core.fruit_db import FRUIT_CATALOG
        shape = {"Twining": "Pole", "Tendril": "Arch", "Clinging": "Wall", "Trailing": "Ground"}
        fr = FRUIT_CATALOG["cucurbita_pepo"].fruit
        r = VineEngine(preset.profile, preset.leaf, preset.venation).generate(
            guide_shape(shape[preset.profile.mode.value], 1.5, 1.5), seed=3, detail=0.5, fruit=fr)
        return [r.stem.vertices, r.tendrils.vertices, r.leaves.vertices, r.fruits.vertices, r.roots.vertices]
    if form == "Flower":
        from core.inflorescence import InflorescenceEngine
        r = InflorescenceEngine(preset.flower, preset.infl).generate(seed=3, detail=0.3, max_flowers=12)
        return [r.mesh.vertices]
    from core.plant_pipeline import BotanicalPlantPipeline
    r = BotanicalPlantPipeline(preset).generate(seed=3, leaf_budget=1500, roots=True, shoot_leaves=0)
    out = [np.concatenate([a.positions for a in r.skeleton_graph.axes]), r.foliage.positions,
           np.asarray(r.leaf_mesh_data["vertices"])]
    if r.root_graph is not None and r.root_graph.axes:
        out.append(np.concatenate([a.positions for a in r.root_graph.axes]))
    return out


def _perturb(value, rng_range):
    if isinstance(value, bool):
        return not value
    if isinstance(value, Enum):
        others = [e for e in type(value) if e != value]
        return others[len(others) // 2]
    if isinstance(value, (int, float)):
        if rng_range:
            lo, hi = rng_range
            far = hi if abs(hi - value) >= abs(value - lo) else lo
            return type(value)(far)
        return type(value)(value * 2 + 3)
    if isinstance(value, tuple):
        return tuple(0.9 if i % 2 == 0 else 0.1 for i in range(len(value)))
    return value


def check(form: str, key: str, report=print) -> list:
    """Returns the inactive fields whose change altered the geometry (should be empty)."""
    base = P._catalog(form)[key]
    ref = _signature(form, base)
    ranges = P._ranges(form)
    failures = []
    for path in inactive_fields(form, base):
        if any(tag in path for tag in MATERIAL_ONLY):
            continue
        section, field = path.split(".")
        obj = copy.deepcopy(base)
        sec = getattr(obj, section)
        setattr(sec, field, _perturb(getattr(sec, field), ranges.get(path)))
        sig = _signature(form, obj)
        same = len(sig) == len(ref) and all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(sig, ref))
        if not same:
            failures.append(path)
            report(f"  CHANGED {form}/{key}: {path}")
    return failures


if __name__ == "__main__":
    import time
    t0 = time.time()
    bad = []
    for form, keys in CASES.items():
        for key in keys:
            n = len(inactive_fields(form, P._catalog(form)[key]))
            print(f"{form}/{key}: {n} inactive fields")
            bad += check(form, key)
    print(f"{len(bad)} rules contradicted by the geometry ({time.time() - t0:.1f} s)")
    sys.exit(1 if bad else 0)
