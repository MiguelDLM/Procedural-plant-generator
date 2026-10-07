"""
Plant Ontology (PO) Integration.
Standardized botanical ontology terms from Planteome / OBO Plant Ontology (plantontology.org)
for biological grounding of plant anatomical entities and traits.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PlantOntologyTerm:
    """Plant Ontology (PO) concept representing a standardized botanical anatomical entity."""
    po_id: str
    name: str
    definition: str
    category: str  # 'plant_structure', 'plant_anatomical_space', 'morphological_trait'
    parent_id: Optional[str] = None


# Canonical dictionary of Plant Ontology concepts used in PPG
PLANT_ONTOLOGY_REGISTRY: dict[str, PlantOntologyTerm] = {
    # Shoot and Stem Architecture
    "PO:0009006": PlantOntologyTerm(
        po_id="PO:0009006",
        name="shoot system",
        definition="A collective plant structure composed of shoot axes and leaves.",
        category="plant_structure"
    ),
    "PO:0025029": PlantOntologyTerm(
        po_id="PO:0025029",
        name="shoot axis",
        definition="A cardinal organ part that is the central axis of a shoot system.",
        category="plant_structure",
        parent_id="PO:0009006"
    ),
    "PO:0009046": PlantOntologyTerm(
        po_id="PO:0009046",
        name="stem",
        definition="A shoot axis that is above ground and produces leaves and buds.",
        category="plant_structure",
        parent_id="PO:0025029"
    ),
    "PO:0004712": PlantOntologyTerm(
        po_id="PO:0004712",
        name="trunk",
        definition="A main stem of a woody plant (tree), characterized by secondary growth.",
        category="plant_structure",
        parent_id="PO:0009046"
    ),
    "PO:0025073": PlantOntologyTerm(
        po_id="PO:0025073",
        name="branch",
        definition="A lateral shoot axis that develops from an axillary bud.",
        category="plant_structure",
        parent_id="PO:0025029"
    ),
    "PO:0000035": PlantOntologyTerm(
        po_id="PO:0000035",
        name="stem node",
        definition="A shoot axis node where one or more leaves, branches, or buds attach.",
        category="plant_structure",
        parent_id="PO:0009046"
    ),
    "PO:0000034": PlantOntologyTerm(
        po_id="PO:0000034",
        name="stem internode",
        definition="The portion of a stem or branch between two consecutive nodes.",
        category="plant_structure",
        parent_id="PO:0009046"
    ),
    "PO:0006339": PlantOntologyTerm(
        po_id="PO:0006339",
        name="axillary bud",
        definition="A bud located in the axil of a leaf, capable of developing into a branch.",
        category="plant_structure"
    ),
    "PO:0006340": PlantOntologyTerm(
        po_id="PO:0006340",
        name="terminal bud",
        definition="A bud located at the apex of a stem or branch that drives primary growth.",
        category="plant_structure"
    ),

    # Leaf Anatomy
    "PO:0025004": PlantOntologyTerm(
        po_id="PO:0025004",
        name="vascular leaf",
        definition="A cardinal organ that is photosynthetically active, typically consisting of lamina and petiole.",
        category="plant_structure"
    ),
    "PO:0020039": PlantOntologyTerm(
        po_id="PO:0020039",
        name="leaf lamina",
        definition="The expanded, flattened blade portion of a vascular leaf.",
        category="plant_structure",
        parent_id="PO:0025004"
    ),
    "PO:0020038": PlantOntologyTerm(
        po_id="PO:0020038",
        name="petiole",
        definition="The stalk that attaches the leaf blade to the stem node.",
        category="plant_structure",
        parent_id="PO:0025004"
    ),
    "PO:0000036": PlantOntologyTerm(
        po_id="PO:0000036",
        name="leaf apex",
        definition="The distal tip of the leaf lamina.",
        category="plant_structure",
        parent_id="PO:0020039"
    ),
    "PO:0020042": PlantOntologyTerm(
        po_id="PO:0020042",
        name="leaf margin",
        definition="The boundary or edge of a leaf lamina (entire, serrate, dentate, lobate).",
        category="plant_structure",
        parent_id="PO:0020039"
    ),
    "PO:0020043": PlantOntologyTerm(
        po_id="PO:0020043",
        name="leaf base",
        definition="The proximal region of the leaf lamina nearest to the petiole insertion.",
        category="plant_structure",
        parent_id="PO:0020039"
    ),

    # Leaf Venation Hierarchy (Matos, Duarte et al. 2025; Runions et al. 2005)
    "PO:0005022": PlantOntologyTerm(
        po_id="PO:0005022",
        name="leaf vein",
        definition="A vascular bundle system within the leaf lamina providing transport and structural support.",
        category="plant_structure",
        parent_id="PO:0020039"
    ),
    "PO:0005023": PlantOntologyTerm(
        po_id="PO:0005023",
        name="primary leaf vein",
        definition="The central, thickest vascular axis of the leaf (midrib) originating at petiole.",
        category="plant_structure",
        parent_id="PO:0005022"
    ),
    "PO:0005024": PlantOntologyTerm(
        po_id="PO:0005024",
        name="secondary leaf vein",
        definition="Lateral vascular branch diverging directly from the primary vein.",
        category="plant_structure",
        parent_id="PO:0005022"
    ),
    "PO:0005025": PlantOntologyTerm(
        po_id="PO:0005025",
        name="minor leaf vein",
        definition="Tertiary and higher-order reticulate veins forming intercostal bridges and areoles.",
        category="plant_structure",
        parent_id="PO:0005022"
    ),
    "PO:0005026": PlantOntologyTerm(
        po_id="PO:0005026",
        name="leaf areole",
        definition="The smallest non-vascular photosynthetic mesophyll area enclosed by minor veins.",
        category="plant_anatomical_space",
        parent_id="PO:0020039"
    ),

    # Tissue and Mechanics
    "PO:0004518": PlantOntologyTerm(
        po_id="PO:0004518",
        name="bark",
        definition="The outer protective layer of woody shoot axes, comprising periderm and secondary phloem.",
        category="plant_structure",
        parent_id="PO:0004712"
    ),
    "PO:0005352": PlantOntologyTerm(
        po_id="PO:0005352",
        name="xylem",
        definition="Lignified vascular water-conducting tissue providing structural stiffness.",
        category="plant_structure"
    ),
}


def get_po_term(po_id: str) -> Optional[PlantOntologyTerm]:
    """Retrieves a standardized Plant Ontology term by its PO identifier."""
    return PLANT_ONTOLOGY_REGISTRY.get(po_id)
