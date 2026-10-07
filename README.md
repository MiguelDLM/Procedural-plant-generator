# Procedural Plant Generator (PPG)

[![Blender](https://img.shields.io/badge/Blender-4.2%2B-orange.svg)](https://www.blender.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Pure Python](https://img.shields.io/badge/Dependencies-NumPy%20only-brightgreen.svg)]()

**Procedural Plant Generator** is an empirical, data-driven botanical 3D plant and tree generation extension for **Blender 4.2+**.

Unlike conventional procedural generators (such as Sapling Tree Gen, modular L-systems, or artistic tree creators) that rely on arbitrary heuristic sliders and visual guesswork, **PPG grounds its mathematical morphogenesis directly in peer-reviewed botanical datasets, allometric scaling laws, and measured morphometrics.**

---

## 🔬 Scientific Foundations & Empirical Datasets

PPG bridges plant phenotyping, forestry remote sensing, and computational botany with procedural computer graphics by integrating the following datasets:

| Domain | Empirical Dataset / Reference | Biological Role in PPG |
| :--- | :--- | :--- |
| **Macro-Allometry & Crown Scaling** | **[TALLO Global Tree Allometry](https://developers.google.com/earth-engine/datasets/catalog/TALLO_TALLO_V1)** *(Jucker et al., 2022; Zenodo: 10.5281/zenodo.6637599)* & **[Awesome-Forests](https://github.com/blutjens/awesome-forests)** | Quantifies Stem Diameter ($DBH$) $\leftrightarrow$ Total Height ($H$), Crown Radius ($CR$), Crown Depth ($CD$), and Leonardo da Vinci pipe model tapering ($r_0^\Delta = \sum r_i^\Delta$). |
| **Branching Architecture** | **Hallé-Oldeman Architectural Classification** *(Hallé, Oldeman & Tomlinson, 1978)* + QSM LiDAR | Rhythmic orthotropic vs plagiotropic growth models (**Rauh**, **Massart**, **Troll**, **Attims**), calibrated branching insertion angles, and phyllotaxis ($137.5^\circ$ golden divergence, decussate, distichous). |
| **Leaf Contours & Morphometrics** | **[LeavesBank Dataset](https://github.com/aaslihanyildirim/LeavesBank-Dataset/)** *(Yildirim et al., 2024)* & **[PlantCLEF 2025](https://www.imageclef.org/PlantCLEF2025)** | 198k+ segmented leaf contours parameterized via **Elliptic Fourier Descriptors (EFD)** and continuous forward-leaning serration/dentation waveforms. |
| **Venation Networks & Biomechanics** | **[Dryad Dataset: Leaf Architecture for 122 Species](https://datadryad.org/dataset/doi:10.5061/dryad.1g1jwsv36)** *(Matos, Boakye, Duarte et al., 2025; ESA Bulletin: 10.1002/bes2.2206)* & **[LVD2021 Vein Segmentation](https://github.com/LeryLee/vein_segmentation)** | Direct measurements of **Vein Length per Area ($VLA$)**, primary-secondary-tertiary vein hierarchies, divergence angles, and mechanical cantilever droop under gravity. |

---

## 🌿 Architectural Growth Models (Hallé-Oldeman)

- **Rauh's Model** (*Quercus*, *Pinus*, *Hevea*): Monopodial trunk with rhythmic branching where lateral shoots are morphologically equivalent to the main leader.
- **Massart's Model** (*Abies*, *Araucaria*, *Taxus*, *Ginkgo*): Monopodial trunk with distinct horizontal plagiotropic branch tiers.
- **Troll's Model** (*Fagus*, *Ulmus*, *Acer*): Sympodial growth where lateral branches initially spread horizontally and subsequently bend upright.
- **Attims' Model** (*Eucalyptus*, *Alnus*): Monopodial continuous growth with diffuse branching and high photosynthetic exposure.

---

## 🍃 Mathematical Morphometrics

### 1. Height-DBH & Crown Scaling (TALLO)
$$H(D) = H_{\max} \cdot \left(1 - \exp\left(-\frac{a \cdot D^b}{H_{\max}}\right)\right)$$
$$CR(D) = c \cdot D^d$$

### 2. Leonardo da Vinci's Branching Rule / Murray Pipe Model
$$r_{\text{parent}}^\Delta = \sum_{i=1}^{n} r_{\text{child}, i}^\Delta$$
Where $\Delta \in [2.1, 2.5]$ accounts for wind-induced mechanical stress (Metzger's uniform stress hypothesis).

### 3. Leaf Boundary Fourier Expansion
A closed continuous leaf silhouette is computed without raster artifacts using Fourier harmonics:
$$x(t) = a_0 + \sum_{n=1}^{N} \left(a_n \cos(nt) + b_n \sin(nt)\right)$$
$$y(t) = c_0 + \sum_{n=1}^{N} \left(c_n \cos(nt) + d_n \sin(nt)\right)$$

### 4. Leaf Margin Serration
$$x_{\text{margin}}(\theta) = x(\theta) + \text{sign}(x) \cdot \text{Envelope}(y) \cdot f_{\text{skewed\_sawtooth}}(N_{\text{teeth}} \cdot \theta)$$

---

## 📦 Installation in Blender 4.2+

This add-on is designed as a **pure Python** extension (NumPy is bundled with Blender). No external wheel compilation or pip installs are required.

1. Download or clone this repository into a folder or zip:
   ```bash
   git clone https://github.com/your-username/procedural-plant-generator
   ```
2. In Blender:
   - Go to `Edit` > `Preferences` > `Add-ons` (or `Get Extensions`).
   - Click the top-right down-arrow menu > `Install from Disk...`.
   - Select the `procedural-plant-generator` folder or zipped archive.
3. Enable **Procedural Plant Generator**.
4. The panel will appear in the **3D Viewport Sidebar (`N` key) > `Plant Gen`**.

---

## 🖥️ User Interface Overview

In the **3D Viewport > Plant Gen** tab:
1. **Botanical Taxon**: Select a species from the curated catalog (*Quercus robur*, *Acer palmatum*, *Betula pendula*, *Fagus sylvatica*, *Pinus sylvestris*, *Eucalyptus globulus*, *Ginkgo biloba*, *Ficus elastica*).
2. **Empirical Trait Inspector**:
   - Inspect live allometric scaling formulas ($H$, $CR$, $\Delta$).
   - Examine Hallé-Oldeman architectural traits.
   - Review leaf morphometrics and vein density ($VLA$).
   - Inspect wood density ($\rho$) and leaf mass per area ($LMA$).
3. **Growth & Allometry Controls**:
   - Adjust **DBH (Stem Diameter)** within species-specific biological bounds; height and crown automatically adjust according to TALLO allometry.
   - Adjust **Foliage Density** and **Random Seed**.
4. **Action Buttons**:
   - **Generate Botanical Plant**: Spawns complete 3D tree/plant with procedural PBR materials.
   - **Generate Macro Leaf (3D Veins)**: Generates a standalone high-resolution leaf with 3D primary and secondary venation.
   - **Export Trait Report**: Exports the full empirical parameters to a structured JSON document for scientific documentation.

---

## 🧪 Running Unit Tests

The test suite validates allometric scaling, Fourier contours, venation graphs, and biomechanics:

```bash
cd procedural-plant-generator
python3 -m unittest discover -s tests -v
```

---

## 📚 References & Acknowledgments

- **TALLO Database**: Jucker, T. et al. (2022). *Tallo: a global tree allometry and crown architecture database.* Global Change Biology.
- **Venation Networks & Mechanics**: Matos, I. S., Boakye, M., Antonio, M., Carlos, S., Chu, A., Duarte, M. A., et al. (2025). *Investigating the Functional and Architectural Diversity of Leaf Venation Networks.* The Bulletin of the Ecological Society of America, 106(1), e02206. Dataset: Dryad [doi:10.5061/dryad.1g1jwsv36](https://doi.org/10.5061/dryad.1g1jwsv36).
- **LeavesBank**: Yildirim, A. et al. (2024). *LeavesBank Dataset: Instance leaf segmentation benchmark.*
- **Leaf Vein Segmentation**: LVD2021 Dataset & HALVS (*Hierarchical Leaf Vein Segmentation*).
- **Awesome-Forests**: Blutjens, L. et al. Curated forest datasets.
