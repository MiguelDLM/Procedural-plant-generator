"""
Botanical Species Preset and Profile Definitions.
Decoupled dataclasses to eliminate circular imports between core and data packages.
"""

from dataclasses import dataclass
from .allometry import AllometricProfile
from .architecture import ArchitectureProfile
from .leaf_morphology import LeafMorphologyProfile
from .leaf_venation import VenationProfile
from .biomechanics import BiomechanicalProfile


@dataclass
class BotanicalSpeciesPreset:
    """Full biological and procedural specification for a plant species."""
    scientific_name: str
    common_name: str
    family: str
    biome: str
    growth_habit: str  # "Canopy Tree", "Sub-canopy Tree", "Shrub", "Herbaceous"

    allometry: AllometricProfile
    architecture: ArchitectureProfile
    leaf_morphology: LeafMorphologyProfile
    venation: VenationProfile
    biomechanics: BiomechanicalProfile
    
    # Plant Ontology & botanical taxonomy metadata
    po_growth_form: str = "PO:0009006"  # shoot system
    notes: str = ""
