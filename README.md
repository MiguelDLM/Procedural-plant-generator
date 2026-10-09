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
| **Bark** | Procedural shader per rhytidome type (Junikka 1994 terminology): fissured, plated, peeling, lenticelled, fibrous, smooth, annulated; each species carries texture descriptors read from field descriptions (Vaucher 2003): blockiness, transverse splits, plate tilt, interlacing, moss and lichen. Fissured bark is an **anastomosing furrow network** (contours of an axially stretched noise) instead of closed cells; blocky barks add Minkowski-Voronoi blocks (exponent 2 irregular .. high rectangular, cracks = F2 − F1) and wavy transverse splits; plates get per-plate tilt and flaky terraces; fibrous barks long strands. Organic warp with high-detail noise whose roughness varies in space and stacked bumps follow Ryan King's procedural bark; per-segment variation follows Substance bark workflows. Bark **ages with girth** (`bark_radius`): smooth periderm with lenticels on thin axes, rhytidome past a species onset radius, fissures widening with diameter (Lefebvre & Neyret 2002; bark-fissure index, MacFarlane & Luo 2009). Colour by depth (inner bark, dirt in cracks, weathered ridges), gentle per-plate tint, cavity AO; moss on upper, shaded and basal surfaces and crustose lichen patches. Optional true displacement (Cycles, adaptive subdivision). | colours, feature size, relief, onset, weathering, blockiness, splits, tilt, interlacing, moss, lichen, displacement |

### Growth forms

A selector at the top of the panel switches between the generators that share materials, caching, the
trait-space arithmetic (blend / variation) and the UI:

| Form | Model | Key parameters |
| :--- | :--- | :--- |
| **Tree / Shrub** | Everything above. | — |
| **Cactus / stem succulent** | Stems are surfaces of revolution around an axis: a generatrix (basal taper, body, superellipse apical dome, optional apical depression) gives columnar, barrel and globose habits. Ribs modulate the cross-section, r = ρ·(1 − depth·(1 − \|cos(mθ′/2)\|<sup>p</sup>)), with optional helical twist; rib numbers default to Fibonacci numbers, as measured in barrel cacti (Robberecht & Nobel 1983). Areoles sit on rib crests (offset by half a step on neighbouring ribs) or, for tuberculate species, on an equal-area spiral lattice at 137.5° (Vogel 1979, generalised from the disc to a surface of revolution); each raises a tubercle (ribs and tubercles as joined vs free podaria, Mauseth 2006). Radial and central spines (gravity curvature, terminal hooks), areolar and apical wool, saguaro/candelabra arms (rise relative to the trunk, outward lean, secondary arms on arms, `crown_fill` placing the branch columns on an area-uniform Vogel spiral over the crown disc so dense crowns have no hollow centre, collision-checked so branches never interpenetrate — e.g. *Pachycereus weberi*'s crown of dozens of branches on a ~2 m trunk), basal offsets, and Opuntia cladode chains in which every daughter pad is tested against the exact volume of all other pads. Weathered epidermis (Evans et al. 1994; Kiesling): paler rib crests and dust in the grooves, vertical streaks, soil splash, dark halos and drip streaks below the areoles (per-vertex, from the nearest areole), corky scars, and patchy epidermal browning — tan to red-orange *scaling* and dark *barking* — rising from the base, higher on the equator-facing side, with isolated crusts ahead of the front. Roots: shallow laterals, optional taproot or napiform storage tuber (peyote), sized from the stem's vascular core, with 90% of roots above a rooting depth (Cannon 1911; Snyman 2005). | habit, height, diameter, dome, ribs (count, depth, sharpness, twist), tubercles, areole spacing, spines, wool, arms, offsets, pads, colours, wax bloom, flecks |
| **Rosette succulent** | Leaves on near-zero internodes at the golden divergence (or distichous), oldest outermost; elevation, size and curvature change with leaf age. Each leaf is a closed volume: bent midline, half-width profile from the lamina descriptors (aspect, widest point, base/apex angles), thickness profile, superellipse cross-section with adaxial channel and abaxial keel; terminal spine and hooked marginal teeth; blushed margins/tips, glaucous bloom, spots and tubercle bands. Clasping, swollen leaf bases and a furled central spike; the stem is a body of revolution (flared root crown, tapering insertion zone) with crescent leaf scars on the phyllotactic spiral, plus persistent withered leaves. Agave leaves carry bud imprints of the neighbouring leaf's teeth and outline, a horny margin and fine striations. Shallow fibrous roots sized by rosette diameter and rooting depth (Franco & Nobel 1990). | leaf count, elevation outer/inner, size gradient, leaf length/aspect/thickness, curvature, section, armature, offsets, colours |

| **Vine / climber** | The stem grows along a **guide path**: a base shape (pole, arch, obelisk spiral, fence, wall, ground run) that can be converted into an editable Bézier curve, or **any curve object of the scene** (one stem per spline, starting at its first point; the vine regenerates while the curve is edited). Climbing modes (Gianoli 2015; Isnard & Silk 2009): *twining* stems coil around the guide as a helix of given radius, pitch and handedness (≈90 % of twiners are right-handed, Edwards et al. 2007; *Humulus* is left-handed); *tendril* climbers follow it and hold on with tendrils — a tendril that has caught the support coils with a **perversion** (handedness reversal) because both ends are fixed (Gerbode et al. 2012), free tendrils coil at the tip, tendrils are axillary, leaf-opposed or replace the terminal leaflets (Sousa-Baena et al. 2018); *clinging* root climbers are pressed to the guide (a wall: its plane is detected from the curve) by adventitious rootlets, with laterals spreading over the support and laminae facing away from it; *trailing* runners creep on the ground, root at the nodes and their fruits rest on the soil. Phytomers along the stem: internodes and leaves expand over a zone behind the apex, then a free searcher tip with an apical hook; lateral shoots droop under their weight. Leaves use the tree leaf engine (all its leaf and venation traits). Parametric fruits: length, diameter, widest point, neck (bottle gourd), ribs, sunken ends (pumpkin), stripes and mottling (watermelon). A *Growth* slider (animatable) sets how much of the guide is covered. | mode, handedness, coil radius/pitch, internode, leaf arrangement, tendrils, laterals, rootlets, fruits, guide shape / curve, growth |

The 15 cactus and 11 rosette presets include several Mexican species (órgano, garambullo, cardón, candelabro, viejito,
biznaga dorada, bonete de obispo, peyote, nopal, maguey, agave azul, *Echeveria*). Algae are out of scope.

### Flowers and inflorescences

Flowers are built from their **floral diagram** (Ijiri et al. 2005; Ronse De Craene 2010) and can be generated alone
(growth form **Flower / Inflorescence**) or placed on any tree, cactus or rosette (**Flowers** panel). Every plant
preset has a default flower (catkins for Fagaceae, Betulaceae, Salicaceae and Juglandaceae; none for conifers).

- **Organs and arrangement.** Calyx, corolla, androecium and gynoecium on a receptacle (ABC model, Coen &
  Meyerowitz 1991). Organs are whorled with merosity *m* (successive whorls alternate by π/m) or spiral at the
  golden angle (*Magnolia*, cactus flowers, double roses). Both come from one generator, as in the sequential
  initiation-with-repulsion model of Kitazawa & Fujimoto (2015). Double flowers add petal whorls in place of stamens.
- **Symmetry.** Zygomorphy is a continuous dorsiventral modulation of organ size, elevation and azimuth along
  d = cos(θ − θ<sub>dorsal</sub>) (Endress 2001): standard-dominated pea flowers (`lip_bias` > 0) or
  lip-dominated bilabiate flowers (`lip_bias` < 0), plus declinate stamens.
- **Petal shape.** The petal outline reuses the leaf half-width model (aspect, widest point, base and apex
  angles and curvatures), written along the proximodistal growth axis (Rolland-Lagan et al. 2003). Extra
  descriptors cover the claw, apex truncation, emargination (cherry) and fringing (*Dianthus*, toothed ligules).
  Posture comes from the opening angle, reflexion (turk's-cap lilies), cup, twist (convolute *Plumeria*) and
  undulation.
- **Sympetaly.** A corolla tube is a surface of revolution r(t) = r₀ + (r₁ − r₀)·t<sup>flare</sup>. With
  `limb_fusion`, each lobe widens to at least its sector π·r/m, so the limb becomes continuous (*Ipomoea*,
  *Petunia*). Stamens and style rise inside the tube following its profile.
- **Capitula.** Disc florets sit on Vogel's spiral r = c√k, θ = 137.508°·k and mature centripetally. Ray
  florets form one rim row or fill the head for double forms (*Dahlia*, *Tagetes*); phyllaries sit beneath.
- **Inflorescences.** Solitary, raceme, spike, catkin, umbel, corymb (pedicels lengthening basipetally to a
  flat top) and panicle, optionally with umbellate laterals (*Agave* scape). Flowers open acropetally, with the
  `Bloom Stage` slider shifting the whole gradient (Prusinkiewicz et al. 2007; Weberling 1989). Flowers are
  built at four opening stages, from closed bud to anthesis, and transformed into place.
- **Placement on plants.** On trees, flowers sit at the tips of the youngest shoots or along them (axillary
  spurs, cauliflory), oriented upright (candles), along the shoot, or pendent (catkins, *Robinia*). On cacti
  they grow from areoles in a ring below each apex, or on the distal margin of Opuntia pads. On rosettes they
  form a terminal scape (*Agave*, *Aloe*) or a few lateral inflorescences from mature leaf axils (*Echeveria*,
  *Dudleya*). One prototype inflorescence is instanced with Geometry Nodes, so thousands of sites cost little.
- **Colour.** Base-to-tip gradient, a contrasting eye, nectar guides, spots (*Lilium*), a tint on outer tepals,
  per-flower value jitter and a velvety sheen.

The 50 flower presets each carry a floral formula. They include Mexican flowers such as dalia, cempasúchil,
mirasol (*Cosmos*), nochebuena, cacaloxóchitl (*Plumeria*), manto de la virgen (*Ipomoea*), and the flowers
of saguaro, nopal, cardón and other cacti, plus the maguey quiote.

The 11 vine presets: *Ipomoea purpurea*, *Phaseolus coccineus* (ayocote), *Humulus lupulus*, *Wisteria
sinensis*, *Pisum sativum*, *Cucumis sativus*, *Cucurbita pepo*, *Citrullus lanatus*, *Vitis vinifera*,
*Hedera helix* and *Passiflora caerulea*. Alpha-mapped leaf cards overlap densely in vines; the add-on raises
Cycles' transparent bounces to 32 when needed (dense foliage otherwise renders black).

### Grasses and cereals

The **Grass / Cereal** growth form (`core/grass.py`) builds grasses phytomer by phytomer, after the
architectural crop models ADEL-Maize / ADEL-Wheat (Fournier & Andrieu 1998), the maize phytomer geometry of
Wen et al. (2021) and the tillering rules of Evers et al. (2005):

- **Tillers** around the main shoot (only a share of them elongate a culm and flower; the others stay leafy).
- **Culm**: internodes lengthening up the stem; one leaf per node, alternating at 180°. The **sheath** wraps
  the culm, the **blade** starts at the ligule; blade length and width follow a bell along the culm (flag
  leaf shorter). Blades are real ribbons, not cards: insertion angle, bending under their weight toward the
  tip, V-fold along the pale midrib, twist, wavy margins (maize), fine parallel veins, dry tips.
- **Inflorescences**: distichous **spikes** with awns (wheat, barley), **panicles** in whorls (oats, rice,
  sorghum), silky **plumes** (pampas and fountain grass), comb-like **one-sided spikes** (blue grama), and in
  maize the **tassel** plus **ears** on shanks: cob with staggered kernel rows, husk leaves (closed, or peeled
  back with *Husk*) and silks.
- **Roots**: fibrous crown roots and maize brace (prop) roots. **Ripeness** turns the plant straw-gold and
  makes heavy heads nod.
- **Lawn / Meadow**: several tuft variants scattered over a patch (tufts per m², minimum spacing, size
  variation) with Geometry Nodes instancing.

Presets: maize, bread wheat, barley, oats, rice, sorghum, sugarcane, perennial ryegrass (lawn), blue grama
(navajita), fountain grass, pampas grass.

### Orchids

The **Orchid** growth form (`core/orchid.py`) follows the floral diagram of Orchidaceae and the growth habits
described in floras and plant patents:

- **Flower**: three sepals (dorsal, two lateral, or a fused **synsepal**) and three petals, the median one
  modified into the **labellum** (its own identity: the "perianth code" of Hsu et al. 2015), plus the
  **column** with anther cap and stigma, a **staminode** (Paphiopedilum) or **mentum** (Dendrobium). Tepals use
  the leaf half-width model with claw, cupping, reflexion, twist and wavy margins.
- **Labellum**: outline as a smooth union of claw, lateral lobes and mid lobe (with isthmus); the lamina bends
  without stretching, so lateral lobes fold up around the column (Phalaenopsis), the lip rolls into a tube
  (Cattleya, Dendrobium, Vanilla) or a **pouch** (Paphiopedilum); callus pad or keels, apical cirrhi, frilled
  margins (sum of sine waves, McCord & Wünsche 2008).
- **Resupination**: the ovary twists until the lip is lowermost, computed as the gravitropic angle needed
  (Rowe et al. 2025); `resupination = 0` keeps the lip uppermost (*Prosthechea cochleata*). The six-ribbed
  ovary shows the twist.
- **Inflorescences** from the leaf axils (3rd–4th leaf below the apex in Phalaenopsis), the apex or base of a
  pseudobulb, or the nodes of old canes. The axis is an elastic cantilever (dθ/ds = M/EI, EI ∝ r⁴, stiff
  peduncle) loaded by its flowers: erect, arching or pendent. Flowers open from the base up and turn to the
  light; capsules (vanilla beans) can replace some flowers.
- **Pigmentation**: full colour, spots and venation (the three Phalaenopsis patterns of PeMYB2/11/12), bars,
  coloured tips, lip throat, all drawn in organ space so spots keep their size.
- **Habit**: monopodial stem with two-ranked fleshy leaves; sympodial growths along a zigzag rhizome with
  pseudobulbs (widest point, fullness, compression, ridges, nodes, papery sheaths, wrinkled leafless
  backbulbs) and leaves at the apex, in a basal fan or along canes; or a climbing vine on a post with one
  clinging root per node opposite the leaf (Vanilla). Thick aerial roots with velamen and a green tip.

Presets (measurements from USPTO plant patents, Flora of China, POWO, Flora of North America): moth orchid
(*Phalaenopsis*), *Cattleya labiata*, *Dendrobium nobile*, *Paphiopedilum insigne*, *Oncidium sphacelatum*,
*Cymbidium* hybrids, *Prosthechea cochleata* (clamshell orchid) and *Vanilla planifolia*.

### Vegetables: root crops, tubers and heads

The **Vegetable** growth form (`core/vegetable.py`) covers the harvested organs that are not fruits:

- **Storage taproots** (carrot, radish, beetroot, turnip): a body of revolution from the crown down — rounded
  shoulder, widest point, conical or rounded taper into a thin tail — partly above the soil with its own
  colour there (green or purple shoulders), white-tipped radishes, growth rings, and lateral rootlets in
  vertical ranks along the xylem poles.
- **Tubers** (potato): leaning leafy stems; stolons from the underground nodes swell into tubers whose eyes sit
  on a ~2/5 spiral crowded toward the rose end, as dimples measured on the surface.
- **Inflorescence heads** (cauliflower, broccoli, Romanesco): the curd is an inflorescence whose meristems keep
  branching in the same golden-angle spiral at every scale (Azpeitia et al. 2021; Kieffer et al. 1998). It is
  built as one continuous dome or cone displaced by nested height fields — every order of meristems in a
  spiral around those of the previous order, each bump grown along its meristem's axis: merged rounded caps
  (cauliflower curd), lobes covered with bud granules over visible branches (broccoli), or cones on cones
  (Romanesco). Crevices are shaded through a height attribute.
- Leaves from the tree leaf engine in a basal rosette or along the stems; brassica inner leaves curl up around
  the head. *Lift* raises the plant to show the underground organs, as when harvested.

Presets: carrot, radish, beetroot, turnip, potato, cauliflower, broccoli, Romanesco.

### Fruits and bunches

One parametric fruit (`core/fruit.py`) covers pomes, drupes, berries, citrus, pomegranates, pepos and pods. Its
outline uses the shape descriptors of Tomato Analyzer (Brewer et al. 2006): length and width, widest point,
blunt or pointed ends, a neck, stalk cavity and calyx basin (dimples around the axis), ribs or lobes and
lopsidedness; the persistent calyx forms a crown at the blossom end (pomegranate, apple, quince) or a star on
the stalk end (tomato). Skin: sun-side blush, streaks or stripes, lenticels / oil glands, russet, waxy bloom,
gloss. Bunches are panicles of berries packed in a cone with shoulders and an adjustable compactness (Tello &
Ibáñez 2018), e.g. grapes; cherries hang in twos and threes on long pedicels.

- **Fruits panel** (Tree and Vine forms): the fruit borne by the plant (each fruit tree and fruiting vine
  preset has a default), fruiting density, scale. On trees, fruits hang from the flowering sites of the
  current inflorescence (terminal or axillary); on vines they hang from the nodes or rest on the soil.
- **Fruit / Bunch** growth form shows one fruit or bunch on its own. 19 fruit presets: apple, pear, quince,
  sweet cherry, peach, olive, grapes (red and white), tomato, pomegranate, orange, lemon, pumpkin,
  watermelon, cucumber, passion fruit and three legume pods. Fruit presets are shared as JSON like the
  others (`growth_form: "Fruit"`), and tree / vine presets may name a `default_fruit`.

### Forests and performance

- **Forest panel** (Tree form): list the species of the mix (each with a number of variants and,
  optionally, the current sliders), then **Create Forest**. Each variant is a unique tree (own seed,
  trait-space mutation, stem diameter drawn around the species value) stored as a collection in
  *PPG Forest Library* (excluded from the view layer; optionally marked as assets). Variants of a
  species share their materials and leaf textures.
- **Scatter with Geometry Nodes**: the *PPG Forest* object carries the *PPG Forest Scatter* node group:
  Poisson-disk points on the selected surface (or a generated ground plane) with density in trees/ha and a
  minimum trunk spacing, a slope limit, random variant, rotation and scale, and a viewport display
  fraction (all trees render). All inputs are editable in the modifier panel, and the library collection
  can be used in any other node setup.
- **Instanced leaves** (default): one leaf or shoot card instanced on every foliage point; renders the same
  as real cards with a fraction of the memory and faster updates. Apply the modifier to export real geometry.
- Example: three species × three variants, a 120 m plot at 200 trees/ha: library in ~5 s, about 0.6 M real
  vertices in the file for hundreds of trees.

### Trait space

`core/trait_space.py` maps ~70 continuous traits to [0, 1] (log scale where they span orders of magnitude) plus colours; categorical traits are carried alongside.

- `blend(a, b, t)` – morphological interpolation between two species (UI: **Variants → Blend**)
- `mutate(p, amount, seed)` – intraspecific variant, each trait perturbed by its typical coefficient of variation (UI: **Variation**)
- `distance`, `nearest_species` – the trait report lists the closest catalogue species.

---

### Forests and performance

Large scenes use instancing at two levels, so a forest of hundreds of trees costs little more than its
handful of unique variants:

- **Instanced leaves** (*Foliage › Instanced Leaves*, on by default): one leaf or leafy-shoot card is
  instanced on every foliage point by a shared *PPG Instancer* Geometry Nodes group; per-leaf colour
  jitter reaches the shader through an *Instancer* attribute. Same render, about a million fewer vertices
  per broadleaf tree and faster live updates. Apply the modifier to get real geometry for export.
- **Shared materials**: bark and leaf materials (and leaf textures) are keyed by their parameters and
  reused by every plant that shares them.
- **Forest panel**: list the species of the mix and how many unique variants each gets (its share of the
  mix). *Build Variants* generates them with their own seed, intraspecific trait variation (trait-space
  mutation) and stem-diameter spread, as collections inside *PPG Forest Library* (excluded from the view
  layer, optionally marked as assets); variants of a species share its materials. *Scatter Forest* adds
  the native **PPG Forest Scatter** node group to a *PPG_Forest* object: Poisson-disk points on the
  selected mesh (or a new ground plane) with density in trees/ha and a minimum spacing, a slope limit,
  random variant, Z rotation and scale, and a viewport display fraction (all trees render). Every input is
  editable in the modifier panel, and the library collection can feed any other Geometry Nodes setup.
  *Create Forest* does both in one click.

Rendering: EEVEE is several times faster for previews and many trees; Cycles is needed for true bark
displacement and gives better translucency on leaves, petals and succulent tissue.

### Presets: create, export, import, share

Every plant can be saved as a JSON preset (*Presets* panel): **Save as New Preset** adds the current sliders
to your personal library (it shows up in the species menu), **Export** writes a shareable file (optionally
only the differences from its base species), **Import** and **Import from URL** validate and install other
people's presets. The format, the [JSON Schema](schemas/ppg-preset.schema.json), a
[field reference](docs/preset-fields.md) with units, ranges and meaning, and a guide to build presets for
new species from botanical data (for people and AI agents) are in [docs/PRESETS.md](docs/PRESETS.md).
Presets can be validated without Blender: `python -m core.presets validate my_plant.json`.

Controls that have no effect in the current context are greyed out with a short note (for example rib
depth for cladodes, the corolla tube for a capitulum, branching for palms), and sections with nothing
applicable are hidden. The rules live in `core/relevance.py`; a perturbation test changes every field
they mark inactive and checks that the generated geometry stays identical.

## Species catalogue (40)

| Group | Species |
| :--- | :--- |
| Fagaceae | *Quercus robur*, *Q. rubra*, *Q. agrifolia*, *Fagus sylvatica*, *Castanea sativa* |
| Sapindaceae | *Acer palmatum*, *A. pseudoplatanus*, *A. saccharum*, *Aesculus hippocastanum* |
| Betulaceae / Salicaceae | *Betula pendula*, *Populus tremula*, *P. nigra* 'Italica', *Salix babylonica* |
| Other broadleaves | *Tilia cordata*, *Platanus × hispanica*, *Liriodendron tulipifera*, *Liquidambar styraciflua*, *Ulmus minor*, *Prunus avium*, *Malus domestica*, *Magnolia grandiflora*, *Ficus elastica*, *Olea europaea*, *Cercis canadensis*, *Eucalyptus globulus*, *Ceiba pentandra* |
| Fruit trees | *Pyrus communis*, *Punica granatum*, *Citrus × sinensis* (plus *Malus*, *Prunus avium*, *Olea* above) |
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
- Ijiri, T., Owada, S., Okabe, M. & Igarashi, T. (2005). Floral diagrams and inflorescences: interactive flower modeling using botanical structural constraints. *ACM Transactions on Graphics* 24(3): 720–726.
- Ronse De Craene, L. P. (2010). *Floral Diagrams: An Aid to Understanding Flower Morphology and Evolution.* Cambridge University Press.
- Weberling, F. (1989). *Morphology of Flowers and Inflorescences.* Cambridge University Press.
- Coen, E. S. & Meyerowitz, E. M. (1991). The war of the whorls: genetic interactions controlling flower development. *Nature* 353: 31–37.
- Endress, P. K. (2001). Evolution of floral symmetry. *Current Opinion in Plant Biology* 4: 86–91.
- Kitazawa, M. S. & Fujimoto, K. (2015). A dynamical phyllotaxis model to determine floral organ number. *PLoS Computational Biology* 11(5): e1004145.
- Rolland-Lagan, A.-G., Bangham, J. A. & Coen, E. (2003). Growth dynamics underlying petal shape and asymmetry. *Nature* 422: 161–163.
- Prusinkiewicz, P., Erasmus, Y., Lane, B., Harder, L. D. & Coen, E. (2007). Evolution and development of inflorescence architectures. *Science* 316: 1452–1456.
- Owens, A., Cieslak, M., Hart, J., Classen-Bockhoff, R. & Prusinkiewicz, P. (2016). Modeling dense inflorescences. *ACM Transactions on Graphics* 35(4): 136.
- Junikka, L. (1994). Survey of English macroscopic bark terminology. *IAWA Journal* 15(1): 3–45.
- Vaucher, H. (2003). *Tree Bark: A Color Guide.* Timber Press.
- MacFarlane, D. W. & Luo, A. (2009). Quantifying tree and forest bark structure with a bark-fissure index. *Canadian Journal of Forest Research* 39(10): 1859–1870. doi:10.1139/X09-098
- King, R. (2022). *Procedural Tree Bark (Blender Tutorial)*. https://www.youtube.com/watch?v=6ECeHoATa74
- Games Artist. *Tree bark material breakdown*. https://gamesartist.co.uk/tree-bark/
- Lefebvre, S. & Neyret, F. (2002). Synthesizing bark. *Eurographics Workshop on Rendering*, 105–116. doi:10.2312/EGWR/EGWR02/105-116
- Federl, P. & Prusinkiewicz, P. (2004). Finite element model of fracture formation on growing surfaces. *ICCS 2004*, LNCS 3037: 138–145.
- Evans, L. S., McKenna, C., Ginocchio, R., Montenegro, G. & Kiesling, R. (1994). Surficial injuries of several cacti of South America. *Environmental and Experimental Botany* 34: 285–292.
- Cannon, W. A. (1911). *The Root Habits of Desert Plants.* Carnegie Institution of Washington, Publ. 131.
- Snyman, H. A. (2005). A case study on in situ rooting profiles and water-use efficiency of cactus pears, *Opuntia ficus-indica* and *O. robusta*. *Journal of the Professional Association for Cactus Development* 7: 1–21.
- Franco, A. C. & Nobel, P. S. (1990). Influences of root distribution and growth on predicted water uptake and interspecific competition. *Oecologia* 82: 151–157. doi:10.1007/BF00323528
- Fournier, C. & Andrieu, B. (1998). A 3D architectural and process-based model of maize development. *Annals of Botany* 81: 233–250. doi:10.1006/anbo.1997.0549
- Wen, W. et al. (2021). 3D phytomer-based geometric modelling method for plants — the case of maize. *AoB Plants* 13: plab055. doi:10.1093/aobpla/plab055
- Evers, J. B. et al. (2005). Towards a generic architectural model of tillering in Gramineae, as exemplified by spring wheat. *New Phytologist* 166: 801–812. doi:10.1111/j.1469-8137.2005.01337.x
- Azpeitia, E. et al. (2021). Cauliflower fractal forms arise from perturbations of floral gene networks. *Science* 373: 192–197. doi:10.1126/science.abg5999
- Kieffer, M., Fuller, M. P. & Jellings, A. J. (1998). Explaining curd and spear geometry in broccoli, cauliflower and 'romanesco': quantitative variation in activity of primary meristems. *Planta* 206: 34–43.
- Brewer, M. T. et al. (2006). Development of a controlled vocabulary and software application to analyze fruit shape variation in tomato and other plant species. *Plant Physiology* 141: 15–25. doi:10.1104/pp.106.077867
- Spjut, R. W. (1994). A systematic treatment of fruit types. *Memoirs of the New York Botanical Garden* 70: 1–182.
- Tello, J. & Ibáñez, J. (2018). What do we know about grapevine bunch compactness? A state-of-the-art review. *Australian Journal of Grape and Wine Research* 24: 6–23. doi:10.1111/ajgw.12310
- Darwin, C. (1875). *The Movements and Habits of Climbing Plants.* John Murray.
- Gianoli, E. (2015). The behavioural ecology of climbing plants. *AoB Plants* 7: plv013. doi:10.1093/aobpla/plv013
- Isnard, S. & Silk, W. K. (2009). Moving with climbing plants from Charles Darwin's time into the 21st century. *American Journal of Botany* 96: 1205–1221.
- Edwards, W., Moles, A. T. & Franks, P. (2007). The global trend in plant twining direction. *Global Ecology and Biogeography* 16: 795–800.
- Gerbode, S. J., Puzey, J. R., McCormick, A. G. & Mahadevan, L. (2012). How the cucumber tendril coils and overwinds. *Science* 337: 1087–1091.
- Sousa-Baena, M. S., Sinha, N. R., Hernandes-Lopes, J. & Lohmann, L. G. (2018). Convergence and divergence in the evolution of tendrils in angiosperms. *Annals of Botany* 122: 1–18.
- Vecchiato, G. et al. (2023). A 2D model to study how secondary growth affects the self-supporting behaviour of climbing plants. *PLoS Computational Biology* 19: e1011538. doi:10.1371/journal.pcbi.1011538
- Hädrich, T., Benes, B., Deussen, O. & Pirk, S. (2017). Interactive modeling and authoring of climbing plants. *Computer Graphics Forum* 36(2): 49–61. doi:10.1111/cgf.13106
- Wang, W., Jüttler, B., Zheng, D. & Liu, Y. (2008). Computation of rotation minimizing frames. *ACM Transactions on Graphics* 27(1): 2.
- Cooper, L. et al. (2018). The Planteome database. *Nucleic Acids Research* 46: D1168–D1180.

---

## License & Commercial Use

This project is licensed under the **GNU General Public License v3.0 (or later)** — see the [LICENSE](LICENSE) file for the full text.

- **Personal, Educational & Academic Use**: Completely free and open. You may study, run, and experiment with the software without restrictions.
- **Modifications & Derivative Works (Copyleft)**: If you modify this software or build derivative tools upon it and distribute them, you must make your modifications open source under the **GNU GPLv3** terms with full attribution.
- **Generated 3D Output**: 3D plant meshes, procedural textures, and renders produced by running the add-on are your own creative work and can be used freely in personal and commercial art, films, or games.
- **Presets & Botanical Data**: Presets in `presets/` are released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

### Commercial / Proprietary Licensing (Dual License)

If you or your organization wish to incorporate Procedural Plant Generator algorithms, code, or derivative tools into a proprietary, closed-source engine, pipeline, or commercial software product where the copyleft requirements of the GNU GPLv3 are not compatible, a separate **Commercial License** can be arranged.

For commercial licensing and custom integration inquiries, contact:  
**Miguel Diaz de Leon-Munoz** — [migueldlm1307@gmail.com](mailto:migueldlm1307@gmail.com)

