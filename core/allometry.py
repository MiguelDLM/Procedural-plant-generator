"""
Botanical Allometry and Scaling Laws.
Based on empirical datasets including:
- TALLO: A global tree allometry dataset (Jucker et al., 2022; Zenodo: 10.5281/zenodo.6637599)
- Awesome-Forests inventories (Auto Arborist, NEON, USFS)
- Metabolic Scaling Theory & Leonardo's rule (West, Brown & Enquist, 1999; Shinozaki et al., 1964)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class AllometricProfile:
    """Empirical scaling parameters for a plant or tree taxon."""
    # Stem diameter at breast height (DBH, at 1.3m) reference and boundaries in meters
    dbh_min_m: float = 0.02
    dbh_max_m: float = 1.20
    dbh_default_m: float = 0.35

    # Height allometry: H = min(h_max, a * (DBH * 100)^b) or asymptotic Weibull
    height_a: float = 1.85      # Scaling coefficient
    height_b: float = 0.58      # Allometric exponent (typically 0.52 - 0.65 in TALLO)
    height_max_m: float = 32.0   # Asymptotic maximum height in meters

    # Crown radius allometry: CR = c * (DBH * 100)^d
    crown_radius_c: float = 0.42
    crown_radius_d: float = 0.53

    # Crown depth ratio: Crown Length / Total Height (typically 0.40 - 0.75)
    crown_depth_ratio: float = 0.55

    # Leonardo da Vinci's rule / Murray pipe model exponent: r_parent^delta = sum(r_children^delta)
    # Pure Leonardo = 2.0; empirical trees under wind load = 2.1 - 2.6
    pipe_exponent_delta: float = 2.25

    # Stem buttress / root flare intensity and decay
    buttress_amplitude: float = 0.45   # Proportional expansion at soil level
    buttress_decay: float = 12.0       # Exponential rate of decay upwards
    buttress_lobes: int = 0            # Gielis symmetry m of the fluted base (0 = round)

    # Wood physical traits
    wood_density_g_cm3: float = 0.62   # Typical angiosperm/gymnosperm wood density
    bark_fraction: float = 0.08        # Radial fraction occupied by bark


class AllometricEngine:
    """Calculates scientifically accurate plant dimensions from DBH and taxon profiles."""

    def __init__(self, profile: AllometricProfile = None):
        self.profile = profile or AllometricProfile()

    def calculate_height(self, dbh_m: float) -> float:
        """
        Calculate total height from DBH (in meters) using TALLO allometric power law
        with asymptotic capping.
        """
        dbh_cm = max(1.0, dbh_m * 100.0)
        h_power = self.profile.height_a * (dbh_cm ** self.profile.height_b)
        # Asymptotic saturation near height_max_m
        h = self.profile.height_max_m * (1.0 - np.exp(-h_power / self.profile.height_max_m))
        return float(np.clip(h, 0.5, self.profile.height_max_m))

    def calculate_crown_radius(self, dbh_m: float) -> float:
        """
        Calculate mean horizontal crown radius (in meters) from DBH.
        """
        dbh_cm = max(1.0, dbh_m * 100.0)
        cr = self.profile.crown_radius_c * (dbh_cm ** self.profile.crown_radius_d)
        return float(np.clip(cr, 0.2, 25.0))

    def calculate_crown_depth(self, total_height_m: float) -> float:
        """
        Calculate vertical length of the crown foliage zone in meters.
        """
        return float(total_height_m * self.profile.crown_depth_ratio)

    def calculate_stem_taper(self, z_relative: np.ndarray, base_radius_m: float) -> np.ndarray:
        """
        Calculates radius along relative stem height z in [0, 1].
        Combines Leonardo/Metzger tapering with basal root flare (buttressing).
        
        Args:
            z_relative: 1D array of height fractions from 0.0 (ground) to 1.0 (stem apex).
            base_radius_m: Radius at DBH/stem base.
        Returns:
            1D array of stem radii in meters.
        """
        z = np.clip(z_relative, 0.0, 1.0)
        delta = self.profile.pipe_exponent_delta

        # Base taper: Metzger uniform mechanical stress profile
        # r(z) = r_base * (1 - z)^(1 / delta)
        taper_base = base_radius_m * ((1.0 - z * 0.96) ** (1.0 / delta))

        # Basal buttress flare near soil surface (decays rapidly above z > 0.15)
        flare = 1.0 + self.profile.buttress_amplitude * np.exp(-self.profile.buttress_decay * z)

        return taper_base * flare

    def split_child_radii(self, parent_radius: float, split_weights: list[float]) -> list[float]:
        """
        Computes child branch radii conserving cross-sectional area / stress
        according to Leonardo's rule:
            r_parent^delta = sum(r_child_i^delta)
        """
        delta = self.profile.pipe_exponent_delta
        weights = np.array(split_weights, dtype=float)
        weights = np.clip(weights, 1e-4, None)
        norm_weights = weights / np.sum(weights)

        # r_child_i = r_parent * (norm_weight_i)^(1 / delta)
        child_radii = [float(parent_radius * (w ** (1.0 / delta))) for w in norm_weights]
        return child_radii
