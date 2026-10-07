# Procedural Plant Generator (PPG)

[![Blender](https://img.shields.io/badge/Blender-4.2%2B%20%7C%205.1%2B-orange.svg)](https://www.blender.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Pure Python](https://img.shields.io/badge/Dependencies-NumPy%20only-brightgreen.svg)]()

**Procedural Plant Generator** is a Blender 4.2+ / 5.x extension that builds trees, cacti and rosette succulents from **measurable botanical traits** instead of artistic sliders. Every species is a point in a normalised *trait space*: allometry, crown envelope, branching architecture, leaf-architecture descriptors, venation, leaf area index and bark. New plants are made by editing traits, blending species, or sampling intraspecific variation.

---

## What is modelled, and how

| Level | Model | Parameters (all editable) |
| :--- | :--- | :--- |
| **Size** | Saturating height–DBH and power-law crown allometry; coefficients are fitted per species from anchor values (DBH, height, crown radius). | DBH, H<sub>max</sub>, height/crown exponents, crown depth ratio |
| **Crown silhouette** | Beta envelope E(u) = u<sup>a</sup>(1−u)<sup>b</sup>/max with a = kp, b = k(1−p): conical, ovoid, spherical, columnar and umbrella crowns are points of one 2-D space (cf. Horn 1971). Laterals are sized so their *horizontal reach* meets the envelope. | widest position p, fullness k |
| **Architecture** | Hallé–Oldeman models (Rauh, Massart, Troll, Attims, Corner); apical dominance separates **excurrent** (single leader) from **decurrent** (codominant leaders) crowns; phyllotaxis (spiral/decussate/distichous/whorled) sets lateral azimuths; plagiotropy flattens sprays; gravitropism/phototropism/tortuosity and cantilever **self-weight bending** are integrated while each axis grows, so children always stay attached. | branch & twig angles, frequency, length ratio, tropisms, whorl size, max order |
| **Wood radii** | Pipe model: an axis cross-section is shared among its laterals in proportion to the foliage they carry, r<sub>i</sub> = R·(L<sub>i</sub>²/ΣL²)<sup>1/Δ</sup>, with near-conical taper to fine tips; Gielis superformula flutes and root flare on the trunk. | Δ, buttress lobes m, flare amplitude/decay, n1–n3 |
| **Leaf outline** | Each foliar unit is a union of **blades** (lamina, lobe, leaflet, needle). A blade's half-width w(t) is two cubic Béziers whose end tangents are the botanical **base and apex angles** (Manual of Leaf Architecture, Ellis et al. 2009). Lobes are smooth-unioned blades (pinnate along the midvein, palmate from the petiole); compound leaves, needle fascicles, conifer sprays and palm fronds use the same primitive. | L/W, widest point, base/apex angle & curvature, cordate depth, asymmetry, apical notch, falcate bend, lobes, margin teeth, leaflets |
| **Margin** | Tooth waveforms per type: serrate, doubly serrate, dentate, crenate, sinuate, spinose. | count, depth, skew |
| **Venation** | Per blade: midvein, secondaries by pattern (craspedodromous, brochidodromous with marginal loops, eucamptodromous, actinodromous with basal primaries, parallelodromous, dichotomous flabellate), percurrent tertiaries, and a Voronoi minor reticulum whose areole width follows **VLA** (d ≈ 2/VLA for a polygonal network). Radii follow a hydraulic hierarchy. | pattern, VLA, secondary pairs/angle/arcuation, reticulation |
| **Leaf texture** | Pure-NumPy raster of colour + alpha + vein height: mesophyll mottling, lighter veins, petiole/twig tissue, and **senescence** that invades from margins and intercostal areas while tissue along veins stays green longest. | colours (adaxial, abaxial, vein, autumn), gloss, season |
| **Foliage** | Cards carry a single leaf or a **leafy shoot** (several leaves in phyllotaxis on a stem, younger leaves smaller). Their number comes from **leaf area index** × crown projection ÷ unit leaf area; their tilt follows the species' **mean leaf angle** (planophile ↔ erectophile). Palms get an apical frond rosette. | LAI, mean leaf angle, leaves per shoot, card budget |
| **Forks & collars** | Every inserted axis starts with nearly its parent's girth and tapers exponentially to its pipe-model radius (branch collar); codominant leaders grow out of the stem top as one swelling crotch. Root-collar flare lives in the skeleton radii, so axes inserted higher see the real girth. | — |
| **Junctions** | Tubes that meet always intersect, so a seam stays visible however well girths match. Three quality levels: **Tubes** (fastest, forests); **Fused** — the thick parts of stem, leaders, limb bases and main roots are united into one closed surface by a voxel level-set union (OpenVDB remesh) plus volume-preserving Laplacian fillets; **Hero** — Fused plus a local "sleeve" at every fine-branch insertion (child base + a curved slab of the parent's surface that rises just above it at the insertion and sinks below it at its borders), remeshed with a voxel proportional to the child's radius and batched by size and 1 m cells, so cost grows with the number of junctions rather than the crown volume. Fused parts hand over to the continuing tubes tangentially (fused end narrows, tube grows in). Fused surfaces are cached per skeleton. | quality, fusion detail, fillet smoothing; Hero: finest branch radius, fillet detail, max junctions |
| **Bark coordinates** | Seam-free 3D bark coordinates (`bark_base + k·bark_along`, metres) on every wood vertex — tubes and fused surfaces alike — so the procedural bark keeps a constant world scale and stays continuous across forks, cuts and root flares; k sets the anisotropy per bark type (fissures along the axis, lenticels around it). UVs are still provided. | — |
| **Roots** | Root-system types after Köstler et al. (1968): taproot, heart, plate, plus buttress and fibrous (palms). Collar flow shared by Leonardo's rule (verified for coarse roots by Oppelt et al. 2001); zone of rapid taper ≈ 2.2 × DBH; lateral/sinker depths sampled from Y(d) = 1 − β<sup>d</sup> with biome β from Jackson et al. (1996, Table 1); maximum depth after Canadell et al. (1996). Near the stem, main laterals have vertically elongated (plank) sections and a concave top edge, so flares and buttresses grow out of the trunk; stem flutes are aligned with them, and buttress ends carry sinkers (Crook et al. 1997). | system, laterals, spread/crown, max depth, β, taproot share, ZRT, sinker spacing, surface exposure, plank, collar height, knees |
| **Bark** | Procedural shader per rhytidome type: fissured, plated, peeling, lenticelled, fibrous, smooth, annulated. | colours, feature size, relief |

### Growth forms

A selector at the top of the panel switches between three generators that share materials, caching, the
trait-space arithmetic (blend / variation) and the UI:

| Form | Model | Key parameters |
| :--- | :--- | :--- |
| **Tree / Shrub** | Everything above. | — |
| **Cactus / stem succulent** | Stems are surfaces of revolution around an axis: a generatrix (basal taper, body, superellipse apical dome, optional apical depression) gives columnar, barrel and globose habits. Ribs modulate the cross-section, r = ρ·(1 − depth·(1 − \|cos(mθ′/2)\|<sup>p</sup>)), with optional helical twist; rib numbers default to Fibonacci numbers, as measured in barrel cacti (Robberecht & Nobel 1983). Areoles sit on rib crests (offset by half a step on neighbouring ribs) or, for tuberculate species, on an equal-area spiral lattice at 137.5° (Vogel 1979, generalised from the disc to a surface of revolution); each raises a tubercle (ribs and tubercles as joined vs free podaria, Mauseth 2006). Radial and central spines (gravity curvature, terminal hooks), areolar and apical wool, saguaro/candelabra arms (rise relative to the trunk, outward lean, secondary arms on arms, `crown_fill` placing the branch columns on an area-uniform Vogel spiral over the crown disc so dense crowns have no hollow centre, collision-checked so branches never interpenetrate — e.g. *Pachycereus weberi*'s crown of dozens of branches on a ~2 m trunk), basal offsets, and Opuntia cladode chains in which every daughter pad is tested against the exact volume of all other pads. Roots: shallow laterals, optional taproot or napiform storage tuber (peyote), sized from the stem's vascular core, with 90% of roots above a rooting depth (Cannon 1911; Snyman 2005). | habit, height, diameter, dome, ribs (count, depth, sharpness, twist), tubercles, areole spacing, spines, wool, arms, offsets, pads, colours, wax bloom, flecks |
| **Rosette succulent** | Leaves on near-zero internodes at the golden divergence (or distichous), oldest outermost; elevation, size and curvature change with leaf age. Each leaf is a closed volume: bent midline, half-width profile from the lamina descriptors (aspect, widest point, base/apex angles), thickness profile, superellipse cross-section with adaxial channel and abaxial keel; terminal spine and hooked marginal teeth; blushed margins/tips, glaucous bloom, spots and tubercle bands. Clasping, swollen leaf bases and a furled central spike; the stem is a body of revolution (flared root crown, tapering insertion zone) with crescent leaf scars on the phyllotactic spiral, plus persistent withered leaves. Agave leaves carry bud imprints of the neighbouring leaf's teeth and outline, a horny margin and fine striations. Shallow fibrous roots sized by rosette diameter and rooting depth (Franco & Nobel 1990). | leaf count, elevation outer/inner, size gradient, leaf length/aspect/thickness, curvature, section, armature, offsets, colours |

The 15 cactus and 11 rosette presets include several Mexican species (órgano, garambullo, cardón, candelabro, viejito,
biznaga dorada, bonete de obispo, peyote, nopal, maguey, agave azul, *Echeveria*). Algae are out of scope.

### Trait space

`core/trait_space.py` maps ~70 continuous traits to [0, 1] (log scale where they span orders of magnitude) plus colours; categorical traits are carried alongside.

- `blend(a, b, t)` – morphological interpolation between two species (UI: **Variants → Blend**)
- `mutate(p, amount, seed)` – intraspecific variant, each trait perturbed by its typical coefficient of variation (UI: **Variation**)
- `distance`, `nearest_species` – the trait report lists the closest catalogue species.

---

## Species catalogue (37)

| Group | Species |
| :--- | :--- |
| Fagaceae | *Quercus robur*, *Q. rubra*, *Q. agrifolia*, *Fagus sylvatica*, *Castanea sativa* |
| Sapindaceae | *Acer palmatum*, *A. pseudoplatanus*, *A. saccharum*, *Aesculus hippocastanum* |
| Betulaceae / Salicaceae | *Betula pendula*, *Populus tremula*, *P. nigra* 'Italica', *Salix babylonica* |
| Other broadleaves | *Tilia cordata*, *Platanus × hispanica*, *Liriodendron tulipifera*, *Liquidambar styraciflua*, *Ulmus minor*, *Prunus avium*, *Malus domestica*, *Magnolia grandiflora*, *Ficus elastica*, *Olea europaea*, *Cercis canadensis*, *Eucalyptus globulus*, *Ceiba pentandra* |
| Compound leaves | *Fraxinus excelsior*, *Juglans regia*, *Robinia pseudoacacia* |
| Gymnosperms | *Ginkgo biloba*, *Pinus sylvestris*, *P. pinea*, *Picea abies*, *Sequoiadendron giganteum*, *Taxodium mucronatum* (ahuehuete), *Cupressus sempervirens* |
| Monocots | *Phoenix canariensis* |

Trait values are typical adult ranges compiled from general botanical references. They are approximations chosen to reproduce each species' habit and leaf form, not statistical fits to a specific dataset. Adding a species only needs measurable anchors; see `species()` in `core/species_db.py`.

---

## Installation (Blender 4.2+ / 5.x)

1. `Edit > Preferences > Get Extensions > ⌄ > Install from Disk…` and select the zipped folder.
2. Enable **Procedural Plant Generator**.
3. Open **3D Viewport > Sidebar (N) > Plant Gen**.

Panels: *Species* · *Variants (Trait Space)* · *Trunk & Allometry* · *Crown & Branching* · *Leaf Shape* · *Venation & Colour* · *Foliage* · *Roots* · *Bark* · *Topology & Shading*. Changing the species loads all of its traits into the sliders; **Allometric Scaling** keeps height and crown consistent with DBH. Live updates are debounced, and leaf textures/materials are regenerated only when the leaf, venation, season or resolution parameters change.

Typical cost on a desktop CPU: skeleton 0.2–0.5 s, meshes < 0.1 s, a 1024 px leaf texture ≈ 0.5 s (cached).

---

## Tests

```bash
cd procedural-plant-generator
python3 -m unittest discover -s tests -v
```

The suite covers allometry, leaf outlines (non-rectangular, lobed sinuses, teeth), venation hierarchy and VLA, textures and senescence, skeleton connectivity, apical dominance, fork collars, root depth distributions, root–stem merging, junction splitting, hand-overs, hero sleeves and seam-free bark coordinates, mesh consistency, trait-space round-trips/blends/mutations, cactus ribs and spiral areole lattices, rosette age gradients, and generation of every catalogue species (trees, cacti and rosettes).

---

## References

- Ellis, B. et al. (2009). *Manual of Leaf Architecture.* Cornell University Press.
- Hallé, F., Oldeman, R. A. A. & Tomlinson, P. B. (1978). *Tropical Trees and Forests: An Architectural Analysis.* Springer.
- Horn, H. S. (1971). *The Adaptive Geometry of Trees.* Princeton University Press.
- Gielis, J. (2003). A generic geometric transformation that unifies a wide range of natural forms. *American Journal of Botany* 90(3): 333–344.
- Runions, A. et al. (2005). Modeling and visualization of leaf venation patterns. *ACM Transactions on Graphics* 24(3): 702–711. (Space colonization is available in `core/space_colonization.py`.)
- Sack, L. & Scoffoni, C. (2013). Leaf venation: structure, function, development, evolution, ecology and applications. *New Phytologist* 198: 983–1000.
- Shinozaki, K. et al. (1964). A quantitative analysis of plant form – the pipe model theory. *Japanese Journal of Ecology* 14.
- Jucker, T. et al. (2022). Tallo: a global tree allometry and crown architecture database. *Global Change Biology* 28: 5254–5268.
- Köstler, J. N., Brückner, E. & Bibelriether, H. (1968). *Die Wurzeln der Waldbäume.* Paul Parey.
- Jackson, R. B. et al. (1996). A global analysis of root distributions for terrestrial biomes. *Oecologia* 108: 389–411. doi:10.1007/BF00333714
- Canadell, J. et al. (1996). Maximum rooting depth of vegetation types at the global scale. *Oecologia* 108: 583–595. doi:10.1007/BF00329030
- Oppelt, A. L., Kurth, W. & Godbold, D. L. (2001). Topology, scaling relations and Leonardo's rule in root systems from African tree species. *Tree Physiology* 21: 117–128. doi:10.1093/treephys/21.2-3.117
- Crook, M. J., Ennos, A. R. & Banks, J. R. (1997). The function of buttress roots. *Journal of Experimental Botany* 48: 1703–1716. doi:10.1093/jxb/48.9.1703
- Danjon, F., Khuder, H. & Stokes, A. (2013). Deep phenotyping of coarse root architecture in *R. pseudoacacia*. *PLoS ONE* 8: e83548. doi:10.1371/journal.pone.0083548
- Vogel, H. (1979). A better way to construct the sunflower head. *Mathematical Biosciences* 44: 179–189. doi:10.1016/0025-5564(79)90080-4
- Robberecht, R. & Nobel, P. S. (1983). A Fibonacci sequence in rib number for a barrel cactus. *Annals of Botany* 51: 153–155. doi:10.1093/oxfordjournals.aob.a086440
- Mauseth, J. D. (2006). Structure–function relationships in highly modified shoots of Cactaceae. *Annals of Botany* 98: 901–926. doi:10.1093/aob/mcl133
- Cannon, W. A. (1911). *The Root Habits of Desert Plants.* Carnegie Institution of Washington, Publ. 131.
- Snyman, H. A. (2005). A case study on in situ rooting profiles and water-use efficiency of cactus pears, *Opuntia ficus-indica* and *O. robusta*. *Journal of the Professional Association for Cactus Development* 7: 1–21.
- Franco, A. C. & Nobel, P. S. (1990). Influences of root distribution and growth on predicted water uptake and interspecific competition. *Oecologia* 82: 151–157. doi:10.1007/BF00323528
- Cooper, L. et al. (2018). The Planteome database. *Nucleic Acids Research* 46: D1168–D1180.
