"""
Bark descriptors used by the procedural bark shader.

Rhytidome patterns follow common field-guide categories; the shader turns them
into node graphs (blender/materials.py).
"""

from dataclasses import dataclass
from enum import Enum


class BarkPattern(str, Enum):
    SMOOTH = "Smooth"            # Fagus, young Acer, Ficus
    FISSURED = "Fissured"        # Quercus, Castanea, Fraxinus: vertical ridges and furrows
    PLATED = "Plated"            # Pinus, Acer pseudoplatanus: flat scales between cracks
    PEELING = "Peeling"          # Platanus, Eucalyptus: patchy exfoliation
    LENTICELLED = "Lenticelled"  # Betula, Prunus: horizontal lenticel dashes
    FIBROUS = "Fibrous"          # Sequoiadendron, Taxodium, Cupressus: stringy vertical fibres
    ANNULATED = "Annulated"      # Palms: leaf-scar rings


@dataclass
class BarkProfile:
    pattern: BarkPattern = BarkPattern.FISSURED
    base_color: tuple = (0.30, 0.25, 0.20)       # sRGB
    secondary_color: tuple = (0.16, 0.13, 0.10)  # Furrows / lenticels / underbark
    feature_scale_m: float = 0.08                # Characteristic spacing of ridges, plates, lenticels
    relief: float = 0.6                          # Bump strength 0..1
    roughness: float = 0.85
