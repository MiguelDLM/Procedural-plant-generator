"""Core botanical modules for Procedural Plant Generator (pure Python + NumPy)."""
from .allometry import AllometricEngine, AllometricProfile
from .architecture import (ArchitectureEngine, ArchitectureProfile, BranchingGraph, HalleOldemanModel,
                           PhyllotaxisType, CrownShape, crown_envelope)
from .leaf_morphology import (LeafMorphologyEngine, LeafMorphologyProfile, LeafArchetype, MarginType, LobeType,
                              CompoundType, LeafShapeBuilder, apply_archetype_template)
from .leaf_venation import VenationEngine, VenationProfile, VenationPattern
from .leaf_texture import LeafTextureEngine, LeafTexture
from .biomechanics import BiomechanicalEngine, BiomechanicalProfile
from .bark import BarkProfile, BarkPattern
from .foliage import FoliageInstances, place_foliage
from .gielis import GielisEngine, GielisProfile, GIELIS_PRESETS
from .space_colonization import SpaceColonizationEngine, SpaceColonizationConfig
from .ontology import PlantOntologyTerm, PLANT_ONTOLOGY_REGISTRY, get_po_term
from .species_preset import BotanicalSpeciesPreset
from .plant_pipeline import BotanicalPlantPipeline, BotanicalPlantResult
from .trait_space import to_vector, from_vector, blend, mutate, distance, nearest_species

__all__ = [
    "AllometricEngine", "AllometricProfile",
    "ArchitectureEngine", "ArchitectureProfile", "BranchingGraph", "HalleOldemanModel", "PhyllotaxisType",
    "CrownShape", "crown_envelope",
    "LeafMorphologyEngine", "LeafMorphologyProfile", "LeafArchetype", "MarginType", "LobeType", "CompoundType",
    "LeafShapeBuilder", "apply_archetype_template",
    "VenationEngine", "VenationProfile", "VenationPattern",
    "LeafTextureEngine", "LeafTexture",
    "BiomechanicalEngine", "BiomechanicalProfile",
    "BarkProfile", "BarkPattern",
    "FoliageInstances", "place_foliage",
    "GielisEngine", "GielisProfile", "GIELIS_PRESETS",
    "SpaceColonizationEngine", "SpaceColonizationConfig",
    "PlantOntologyTerm", "PLANT_ONTOLOGY_REGISTRY", "get_po_term",
    "BotanicalSpeciesPreset",
    "BotanicalPlantPipeline", "BotanicalPlantResult",
    "to_vector", "from_vector", "blend", "mutate", "distance", "nearest_species",
]
