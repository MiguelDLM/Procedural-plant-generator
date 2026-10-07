# Procedural Plant Generator (PPG)

[![Blender](https://img.shields.io/badge/Blender-4.2%2B%20%7C%205.1%2B-orange.svg)](https://www.blender.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Pure Python](https://img.shields.io/badge/Dependencies-NumPy%20only-brightgreen.svg)]()
[![Plant Ontology](https://img.shields.io/badge/Plant%20Ontology-Planteome%20Standard-success.svg)](https://plantontology.org/)

**Procedural Plant Generator** is an empirical, data-driven botanical 3D plant and tree generation extension for **Blender 4.2+ and Blender 5.1+**.

Unlike conventional procedural generators (such as Sapling Tree Gen or generic L-systems) that rely on arbitrary heuristic sliders and visual guesswork, **PPG grounds its mathematical morphogenesis directly in peer-reviewed botanical datasets, allometric scaling laws, biomechanics, the Plant Ontology, and computational botany algorithms.**

---

## 🔬 Scientific Foundations & Empirical Datasets

PPG bridges plant phenotyping, forestry remote sensing, and computational botany with procedural computer graphics by integrating the following datasets and foundational papers:

| Domain | Empirical Dataset / Scientific Literature | Biological & Algorithmic Role in PPG |
| :--- | :--- | :--- |
| **Macro-Allometry & Crown Scaling** | **[TALLO Global Tree Allometry](https://developers.google.com/earth-engine/datasets/catalog/TALLO_TALLO_V1)** *(Jucker et al., 2022; Zenodo: 10.5281/zenodo.6637599)* & **[Awesome-Forests](https://github.com/blutjens/awesome-forests)** | Quantifies Stem Diameter ($DBH$) $\leftrightarrow$ Total Height ($H$), Crown Radius ($CR$), Crown Depth ($CD$), and Leonardo da Vinci pipe model tapering ($r_0^\Delta = \sum r_i^\Delta$). |
| **Trunk Cross-Sections & Buttressing** | **Gielis Superformula** *(Gielis, 2003; Modular Tree)* | Generates fluted trunk cross-sections, multi-lobed basal buttresses ($m=3, 4, 5, 6, 8$), and elliptical reaction-wood stems using the generalized superformula. |
| **Branching Architecture** | **Hallé-Oldeman Architectural Classification** *(Hallé, Oldeman & Tomlinson, 1978)* + QSM LiDAR / Tree-Gen / UitTree | Rhythmic orthotropic vs plagiotropic growth models (**Rauh**, **Massart**, **Troll**, **Attims**), calibrated branching insertion angles, and phyllotaxis ($137.5^\circ$ golden divergence, decussate, distichous). |
| **Leaf Contours & Morphometrics** | **[LeavesBank Dataset](https://github.com/aaslihanyildirim/LeavesBank-Dataset/)** *(Yildirim et al., 2024)* & **[PlantCLEF 2025](https://www.imageclef.org/PlantCLEF2025)** | 198k+ segmented leaf contours parameterized via **Elliptic Fourier Descriptors (EFD)** and continuous forward-leaning serration/dentation waveforms. |
| **Venation Networks & Auxin Simulation** | **[Runions et al., 2005](https://doi.org/10.1145/1073204.1073251)** & **[Dryad Dataset: Leaf Architecture for 122 Species](https://datadryad.org/dataset/doi:10.5061/dryad.1g1jwsv36)** *(Matos, Boakye, Duarte et al., 2025; ESA Bulletin: 10.1002/bes2.2206)* | **Space Colonization Algorithm** driven by auxin attractors within the lamina, combined with empirical measurements of **Vein Length per Area ($VLA$)**, primary-secondary hierarchies, Murray's hydraulic scaling, and physical cantilever droop under gravity. |
| **Standardized Terminology & Anatomy** | **[Plant Ontology (PO)](https://plantontology.org/)** *(Planteome / OBO)* | Standardized anatomical anchoring across shoot systems (`PO:0009006`), stems (`PO:0009046`), leaf lamina (`PO:0020039`), leaf veins (`PO:0005022`), and areoles (`PO:0005026`). |

---

## 📐 Mathematical Formulations

### 1. Gielis Superformula (2003) for Stem Cross-Sections & Buttresses
$$r(\theta) = \left( \left|\frac{1}{a} \cos\left(\frac{m \theta}{4}\right)\right|^{n_2} + \left|\frac{1}{b} \sin\left(\frac{m \theta}{4}\right)\right|^{n_3} \right)^{-\frac{1}{n_1}}$$
Near ground level ($z < 0.2$), Gielis fluted buttresses ($m=3$ to $8$) produce biological buttressed flares as seen in giant rainforest and montane trees (*Ficus*, *Sequoiadendron*).

### 2. Space Colonization Auxin Venation (Runions et al., 2005)
Auxin hormone attractors $S$ within the leaf blade exert directional pull on nearby vein nodes $V$ within perception distance $d_I$:
$$\vec{D}_v = \frac{\sum_{s \in S_v} \frac{s - v}{\|s - v\|}}{\left\| \sum_{s \in S_v} \frac{s - v}{\|s - v\|} \right\|}$$
Vein segments advance by step size $\delta$ and consume sources within kill distance $d_K$. Segment radii scale according to Murray's hydraulic law:
$$r_v = r_{\text{base}} \cdot \left(\text{flux}_v\right)^{1 / \gamma} \quad (\gamma \approx 2.6 - 3.0)$$

### 3. Height-DBH & Crown Scaling (TALLO)
$$H(D) = H_{\max} \cdot \left(1 - \exp\left(-\frac{a \cdot D^b}{H_{\max}}\right)\right)$$
$$CR(D) = c \cdot D^d$$

### 4. Leonardo da Vinci's Pipe Model
$$r_{\text{parent}}^\Delta = \sum_{i=1}^{n} r_{\text{child}, i}^\Delta \quad (\Delta \approx 2.1 - 2.5)$$

---

## 🌳 Real-World Botanical Presets

PPG provides real-world tree presets with full biological traits and Plant Ontology metadata:

1. **Quercus robur** (*English Oak*): Massive canopy, high wood density (0.71 g/cm³), Rauh architecture, lobed leaves, craspedodromous venation ($VLA = 7.4$ mm/mm²).
2. **Quercus agrifolia** (*Coast Live Oak*): Gnarled crooked branches, convex spiny-dentate leaves, high sclerophylly ($VLA = 9.8$ mm/mm²).
3. **Acer palmatum** (*Japanese Maple*): Decussate branching, 5-lobed palmate leaves, actinodromous primary veins ($VLA = 8.8$ mm/mm²).
4. **Acer pseudoplatanus** (*Sycamore Maple*): Robust canopy maple, Rauh model, serrate palmate lamina ($VLA = 7.6$ mm/mm²).
5. **Betula pendula** (*Silver Birch*): Pendulous weeping twigs (positive gravitropism), doubly serrate ovate leaves, craspedodromous veins ($VLA = 9.6$ mm/mm²).
6. **Fagus sylvatica** (*European Beech*): Troll sympodial architecture with planar zig-zag foliage sprays, high shade tolerance ($VLA = 6.8$ mm/mm²).
7. **Ficus elastica** (*Rubber Fig*): Massive fluted Gielis buttress roots ($m=5$), thick leathery leaves, brochidodromous looping veins ($VLA = 11.2$ mm/mm²).
8. **Pinus sylvestris** (*Scots Pine*): Massart tiered whorls, acicular needle foliage, conifer allometry ($VLA = 2.8$ mm/mm²).
9. **Sequoiadendron giganteum** (*Giant Sequoia*): Massive DBH (up to 4.5m), Gielis $m=8$ fluted buttress base, conical crown, thick bark.
10. **Ginkgo biloba** (*Maidenhair Tree*): Ancient gymnosperm, fan-shaped flabellate leaves with open dichotomous venation.
11. **Eucalyptus globulus** (*Tasmanian Blue Gum*): Attims monopodial continuous architecture, falcate-lanceolate leaves, high vein density ($VLA = 12.4$ mm/mm²).
12. **Populus tremula** (*Common Aspen*): Rapid pioneer allometry, aerodynamic flattened flutter petiole, crenate leaves ($VLA = 8.2$ mm/mm²).

---

## 📦 Installation in Blender 4.2+ / 5.1+

This add-on is a **pure Python** extension (NumPy is bundled with Blender).

1. In Blender:
   - Go to `Edit` > `Preferences` > `Add-ons` (or `Get Extensions`).
   - Click the top-right down-arrow menu > `Install from Disk...`.
   - Select the `procedural-plant-generator` folder or zipped archive.
2. Enable **Procedural Plant Generator**.
3. Access the tool in the **3D Viewport Sidebar (`N` key) > `Plant Gen`**.

---

## 🧪 Unit Tests

Run the test suite verifying allometry, Gielis superformula, Space Colonization, and Plant Ontology:

```bash
cd procedural-plant-generator
python3 -m unittest discover -s tests -v
# 7/7 tests pass in 0.31s
```

---

## 📚 References & Literature

- **Gielis Superformula**: Gielis, J. (2003). *A generic geometric transformation that unifies a wide range of natural forms.* American Journal of Botany, 90(3), 333–344.
- **Space Colonization Leaf Venation**: Runions, A., Fuhrer, M., Lane, B., Federl, P., Rolland-Lagan, A.-G., & Prusinkiewicz, P. (2005). *Modeling and visualization of leaf venation patterns.* ACM Transactions on Graphics (SIGGRAPH 2005), 24(3), 702–711.
- **Venation Architecture & Mechanics**: Matos, I. S., Boakye, M., Antonio, M., Carlos, S., Chu, A., Duarte, M. A., et al. (2025). *Investigating the Functional and Architectural Diversity of Leaf Venation Networks.* The Bulletin of the Ecological Society of America, 106(1), e02206. Dryad DOI: [10.5061/dryad.1g1jwsv36](https://doi.org/10.5061/dryad.1g1jwsv36).
- **TALLO Global Tree Allometry**: Jucker, T. et al. (2022). *Tallo: a global tree allometry and crown architecture database.* Global Change Biology.
- **Plant Ontology**: Cooper, L. et al. (2018). *The Planteome database: an integrated resource for reference ontologies and annotations.* Nucleic Acids Research.
- **Modular Tree**: Blender extension incorporating Gielis cross-sections and modular node-based branching.
