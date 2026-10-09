"""
Wind response of a plant from its height (used by blender.wind).

- First-mode frequency: f ~ D / H^2 for a cantilever; with elastic similarity D ~ H^1.5 (McMahon 1973)
  f ~ H^-0.5, about 0.6 Hz for a 20 m tree and 2-3 Hz for a cereal (de Langre 2008, Annu. Rev. Fluid Mech.
  40: 141).
- Tip deflection grows as v^1.3 rather than v^2 because leaves and crowns reconfigure in the flow (Vogel
  1989, J. Exp. Bot. 40: 941; Vogel number about -0.7).
"""

import math


def wind_frequency(height_m: float) -> float:
    """First-mode frequency (Hz) from the plant height (elastic similarity)."""
    return 0.6 * math.sqrt(20.0 / max(height_m, 0.05))


def wind_amplitude(speed: float, height_m: float) -> float:
    """Tip deflection (m) for a mean wind speed (m/s)."""
    return 0.05 * max(height_m, 0.05) ** 0.75 * (max(speed, 0.0) / 10.0) ** 1.3
