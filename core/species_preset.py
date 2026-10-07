"""
Botanical Species Preset and Profile Definitions.
Decoupled dataclasses to eliminate circular imports between core and data packages.
"""

from dataclasses import dataclass, field
from .allometry import AllometricProfile
from .architecture import ArchitectureProfile
from .leaf_morphology import LeafMorphologyProfile
from .leaf_venation import VenationProfile
from .biomechanics import BiomechanicalProfile
from .bark import BarkProfile
from .roots import RootProfile


@dataclass
class BotanicalSpeciesPreset:
    """Full biological and procedural specification for a plant species."""
    scientific_name: str
    common_name: str
    family: str
    biome: str
    growth_habit: str  # "Canopy Tree", "Sub-canopy Tree", "Shrub", "Palm", ...

    allometry: AllometricProfile
    architecture: ArchitectureProfile
    leaf_morphology: LeafMorphologyProfile
    venation: VenationProfile
    biomechanics: BiomechanicalProfile
    bark: BarkProfile = field(default_factory=BarkProfile)
    roots: RootProfile = field(default_factory=RootProfile)

    # Plant Ontology & taxonomy metadata
    po_growth_form: str = "PO:0009006"  # shoot system
    notes: str = ""
    deciduous: bool = True
