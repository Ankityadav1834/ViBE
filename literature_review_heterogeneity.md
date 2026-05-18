# Literature review: heterogeneity case for 5S5P pack simulation

This note records the assumptions used by `test_5s5p_heterogeneity.py`.

## Why resistance mismatch?

Parallel-connected lithium-ion cells do not automatically share current equally when their ohmic resistances differ. Gogoana, Pinson, Bazant, and Sarma report that a 20% cell internal-resistance mismatch in two parallel cells cycled at 4.5C caused about a 40% cycle-life reduction, and they identify uneven current sharing, higher temperature, and SEI-driven aging as the mechanism.

Hosseinzadeh et al. studied a 1S15P parallel module and found that interconnection resistance and cell count dominate cell-to-cell variation. Their abstract reports that 25% resistance variation leads to 22% current dispersion, while a 30 C thermal gradient leads to 24% current variation.

Implementation choice:

- The model's nominal initial series resistance is about `5.71 mOhm`.
- A 20% mismatch is represented as `R_contact = 0.00114 Ohm`.
- A stronger local weak-link cell uses 30% added resistance, `R_contact = 0.00171 Ohm`.
- The mismatch is applied to the same parallel branch across all five series groups, with one center cell made worse.

Sources:

- Gogoana et al., "Internal resistance matching for parallel-connected lithium-ion cells and impacts on battery pack cycle life", Journal of Power Sources, 2014. https://doi.org/10.1016/j.jpowsour.2013.11.101
- Hosseinzadeh et al., "Quantifying cell-to-cell variations of a parallel battery module for different pack configurations", Applied Energy, 2021; author manuscript: https://wrap.warwick.ac.uk/id/eprint/142117/1/WRAP-quantifying-cell-to-cell-variations-parallel-battery-module-different-pack-configurations-Marco-2020.pdf

## Why cooling heterogeneity?

Thermal boundary conditions in packs vary strongly with cell position, coolant contact, air gaps, tab/busbar layout, and fixture pressure. PyBaMM's Chen2020 parameter set uses a total heat transfer coefficient of `10 W m-2 K-1` and a cell cooling area of `0.00531 m2`, which corresponds to `hA = 0.0531 W/K`. Thermal studies commonly compare weak natural convection around `5-10 W m-2 K-1` against stronger forced/liquid cooling cases; one EV package study reports large temperature rises at 2C-4C discharge, and another prismatic-cell study notes that natural convection at `5-10 W m-2 K-1` has limited cooling impact.

Implementation choice:

- Well-cooled cells use `hA = 0.531 W/K`, equivalent to `h = 100 W m-2 K-1` with the Chen2020 area.
- The weakly cooled branch uses `hA = 0.0531 W/K`, equivalent to `h = 10 W m-2 K-1`.
- This creates a 10x cooling contrast without inventing a new thermal model.

Sources:

- PyBaMM Chen2020 parameter listing: total heat transfer coefficient `10 W m-2 K-1`, cell cooling area `0.00531 m2`, nominal cell capacity `5 Ah`, and SEI parameters. https://docs.pybamm.org/en/stable/source/examples/notebooks/getting_started/tutorial-4-setting-parameter-values.html
- Wu, Lv, and Chen, "Determination of the Optimum Heat Transfer Coefficient and Temperature Rise Analysis for a Lithium-Ion Battery...", Energies, 2017. https://www.mdpi.com/1996-1073/10/11/1723
- "A Physical-Based Electro-Thermal Model for a Prismatic LFP Lithium-Ion Cell Thermal Analysis", Energies, 2025, noting natural convection around `5-10 W m-2 K-1`. https://www.mdpi.com/1996-1073/18/5/1281

## Simulation experiment

The script sets up a 5S5P pack with 5 Ah Chen2020-like cells and the existing reaction-limited SEI model. The heterogeneous branch is intentionally both higher-resistance and weakly cooled, so the diagnostics should show:

- unequal branch currents in the parallel groups,
- higher peak temperature in the weakly cooled branch,
- higher SEI-thickness growth in the stressed cells over cycling.

Run:

```powershell
python test_5s5p_heterogeneity.py
```

For a quick smoke test:

```powershell
python test_5s5p_heterogeneity.py --smoke
```
