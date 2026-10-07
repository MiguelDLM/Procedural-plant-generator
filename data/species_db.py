"""
Re-export botanical species database from core.species_db.
"""

try:
    from ..core.species_db import (
        SPECIES_CATALOG,
        BotanicalSpeciesPreset,
        get_preset_names,
        get_species_preset,
    )
except (ImportError, ValueError):
    from core.species_db import (
        SPECIES_CATALOG,
        BotanicalSpeciesPreset,
        get_preset_names,
        get_species_preset,
    )

__all__ = [
    "SPECIES_CATALOG",
    "BotanicalSpeciesPreset",
    "get_preset_names",
    "get_species_preset",
]
