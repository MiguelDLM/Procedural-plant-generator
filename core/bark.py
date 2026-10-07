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
    # Age: thin axes keep the smooth young periderm (with lenticels); the rhytidome pattern appears
    # once the axis is thicker than `onset_radius_cm` and its fissures widen as it keeps growing in
    # girth (fractures opened by circumferential growth, Lefebvre & Neyret 2002).
    young_color: tuple = None                    # Periderm of young shoots (sRGB); derived when None
    onset_radius_cm: float = 4.0
    inner_color: tuple = None                    # Exposed inner bark at the bottom of fissures
    weathering: float = 0.35                     # Grey, bleached ridge tops

    def __post_init__(self):
        mix = lambda a, b, t: tuple(round((1 - t) * x + t * y, 3) for x, y in zip(a, b))  # noqa: E731
        if self.young_color is None:
            self.young_color = mix(self.base_color, (0.42, 0.40, 0.33), 0.5)
        if self.inner_color is None:
            self.inner_color = mix(self.secondary_color, (0.30, 0.12, 0.06), 0.35)
