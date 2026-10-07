"""
Botanical Biomechanics and Structural Deflection.
Calculates realistic physical drooping of branches and leaves under gravity and self-weight
using beam theory and empirical wood density measurements.
"""

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class BiomechanicalProfile:
    """Mechanical material properties for plant stems and foliage."""
    wood_density_g_cm3: float = 0.62     # Wood dry density (0.15 - 1.25 g/cm^3)
    youngs_modulus_gpa: float = 11.5     # Elastic modulus along grain (4 - 18 GPa)
    leaf_mass_per_area_g_m2: float = 85.0 # LMA from Dryad UCBG 122spp dataset (30 - 250 g/m^2)
    drag_coefficient_cd: float = 0.85    # Aerodynamic drag coefficient
    wind_speed_m_s: float = 0.0          # Ambient wind speed (m/s)


class BiomechanicalEngine:
    """Calculates branch and leaf elastic deformations."""

    def __init__(self, profile: BiomechanicalProfile = None):
        self.profile = profile or BiomechanicalProfile()

    def calculate_branch_tip_droop(self, length_m: float, mean_radius_m: float) -> float:
        """
        Calculates downward tip deflection delta (meters) of a horizontal branch
        under its own self-weight using cantilever beam mechanics:
            delta = (q * L^4) / (8 * E * I)
        where:
            q = weight per unit length = rho_wood * pi * r^2 * g (N/m)
            I = second moment of area = (pi / 4) * r^4 (m^4)
            E = Young's modulus (Pa)
        """
        if length_m <= 0.05 or mean_radius_m <= 0.001:
            return 0.0

        g = 9.81
        rho_kg_m3 = self.profile.wood_density_g_cm3 * 1000.0  # kg/m^3
        E_pa = self.profile.youngs_modulus_gpa * 1e9          # N/m^2

        q = rho_kg_m3 * math.pi * (mean_radius_m ** 2) * g
        I = (math.pi / 4.0) * (mean_radius_m ** 4)

        if E_pa * I <= 1e-6:
            return 0.0

        deflection_m = (q * (length_m ** 4)) / (8.0 * E_pa * I)
        # Cap deflection physically so branches don't bend through themselves
        return float(np.clip(deflection_m, 0.0, length_m * 0.45))

    def calculate_wind_sway_vector(self, height_m: float, wind_dir: np.ndarray = None) -> np.ndarray:
        """
        Calculates static stem displacement from ambient wind drag at height z.
        """
        if self.profile.wind_speed_m_s <= 0.0:
            return np.zeros(3)

        if wind_dir is None:
            wind_dir = np.array([1.0, 0.0, 0.0])
        wind_dir = wind_dir / np.linalg.norm(wind_dir)

        # Wind speed profile increases with height: v(z) ~ v0 * (z/H)^0.2
        rho_air = 1.225
        v = self.profile.wind_speed_m_s * ((height_m / 10.0) ** 0.2)
        drag_pressure = 0.5 * rho_air * self.profile.drag_coefficient_cd * (v ** 2)

        # Flexural deflection increases quadratically with height
        disp_magnitude = drag_pressure * 0.00008 * (height_m ** 2)
        return wind_dir * float(disp_magnitude)
