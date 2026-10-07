"""
Gielis Superformula Engine (Gielis, 2003).
Reference:
Johan Gielis. "A generic geometric transformation that unifies a wide range of natural forms."
American Journal of Botany 90(3): 333–344 (2003).

Provides parametric cross-sections for realistic trunk buttressing, stem fluting,
ribbed bark ridges, and mathematical leaf lamina boundaries as used in Modular Tree.
"""

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class GielisProfile:
    """Parameters for the Gielis Superformula:
    r(theta) = [ |cos(m*theta/4)/a|^n2 + |sin(m*theta/4)/b|^n3 ] ^ (-1/n1)
    """
    m: float = 0.0      # Rotational symmetry (0=circle, 3=triangle, 5=5-ribbed, 6=6-buttress)
    n1: float = 1.0     # Overall curvature / sharpness exponent
    n2: float = 1.0     # Cosine term exponent
    n3: float = 1.0     # Sine term exponent
    a: float = 1.0      # Scaling along major axis
    b: float = 1.0      # Scaling along minor axis


# Canonical Gielis profiles for biological tree structures
GIELIS_PRESETS = {
    # Standard cylindrical stem
    "circular": GielisProfile(m=0.0, n1=1.0, n2=1.0, n3=1.0, a=1.0, b=1.0),
    # Slight elliptical trunk cross-section (reaction wood to prevailing wind)
    "elliptical": GielisProfile(m=2.0, n1=2.0, n2=2.0, n3=2.0, a=1.15, b=0.90),
    # 3-lobed fluted trunk (Carpinus, Taxodium)
    "fluted_3": GielisProfile(m=3.0, n1=0.6, n2=1.4, n3=1.4, a=1.0, b=1.0),
    # 4-fluted buttress
    "buttress_4": GielisProfile(m=4.0, n1=0.45, n2=1.8, n3=1.8, a=1.0, b=1.0),
    # 5-pointed star / palmate leaf / 5-ribbed trunk (Ficus, Acer)
    "star_5": GielisProfile(m=5.0, n1=0.40, n2=1.6, n3=1.6, a=1.0, b=1.0),
    # 6-buttressed tropical rainforest tree base (Ceiba, Ficus)
    "buttress_6": GielisProfile(m=6.0, n1=0.35, n2=2.2, n3=2.2, a=1.0, b=1.0),
    # 8-buttressed massive fluted root base (Sequoiadendron)
    "buttress_8": GielisProfile(m=8.0, n1=0.30, n2=2.5, n3=2.5, a=1.0, b=1.0),
}


class GielisEngine:
    """Evaluates the Gielis Superformula for botanical cross-sections and outlines."""

    @staticmethod
    def evaluate(theta: np.ndarray, profile: GielisProfile) -> np.ndarray:
        """
        Calculates radius r(theta) for an array of angles in radians.
        """
        if profile.m <= 0.0:
            # Degenerate case: pure circle
            return np.ones_like(theta, dtype=float)

        phi = profile.m * theta / 4.0

        # Term 1: |cos(phi)/a|^n2
        cos_val = np.abs(np.cos(phi)) / max(1e-5, profile.a)
        term1 = cos_val ** profile.n2

        # Term 2: |sin(phi)/b|^n3
        sin_val = np.abs(np.sin(phi)) / max(1e-5, profile.b)
        term2 = sin_val ** profile.n3

        sum_terms = term1 + term2
        sum_terms = np.clip(sum_terms, 1e-8, None)

        # r = sum_terms ^ (-1 / n1)
        inv_n1 = -1.0 / max(1e-5, profile.n1)
        r = sum_terms ** inv_n1

        # Normalize so average radius is approximately 1.0
        mean_r = np.mean(r)
        if mean_r > 1e-5:
            r = r / mean_r

        return r

    @staticmethod
    def compute_buttressed_cross_section(
        angles: np.ndarray,
        base_radius: float,
        z_relative: float,
        gielis_profile: GielisProfile = None
    ) -> np.ndarray:
        """
        Blends between Gielis fluted buttresses near soil level (z < 0.15)
        and circular cross-section higher up the stem.
        
        Args:
            angles: 1D array of angles around stem perimeter in radians.
            base_radius: Mean stem radius at this height in meters.
            z_relative: Relative height along stem (0.0=ground, 1.0=apex).
            gielis_profile: Desired buttress profile.
        Returns:
            1D array of radii at each angle in meters.
        """
        if gielis_profile is None or gielis_profile.m <= 0.0:
            return np.full_like(angles, base_radius, dtype=float)

        # Buttressing decays exponentially above ground level
        buttress_weight = float(np.exp(-14.0 * z_relative))

        gielis_r = GielisEngine.evaluate(angles, gielis_profile)
        # Linear blend between circular (1.0) and Gielis factor
        modulated_factor = 1.0 + (gielis_r - 1.0) * buttress_weight
        return base_radius * np.clip(modulated_factor, 0.2, 3.5)
