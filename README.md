# Code for *Modeling, Analysis, Optimization and Control for Problems on Networks*

Günter Leugering and Jan Sokołowski

This repository contains the Python scripts that produce the numerical examples,
figures and tables of the book. Each script is self-contained, writes its output
into a folder `results_*` next to itself, and prints the numbers quoted in the text.

## Structure and correspondence with the book

| Chapter | Script | Output folder | Used in |
|---|---|---|---|
| 3 Control problems | `code/ch03_control/inverse_triangle_with_3_leaves.py` | `results_inverse_triangle_with_3_leaves` | identifiability example (triangle with three leaves), case table |
| 3 | `code/ch03_control/tensegrity24_shape_control_explorer.py` | `results_tensegrity24_shape_control` | shape control of the 6-bar/24-string tensegrity |
| 4 Optimization of network structure | `code/ch04_structure_optimization/three_layer_truss_shape_optimization.py` | `three_layer_truss_shape_results` | shape optimization of a three-layer truss |
| 4 | `code/ch04_structure_optimization/cantilever_shape_optimization_nonlinear.py` | `cantilever_nonlinear_shape_results` | geometrically nonlinear cantilever spring network |
| 4 | `code/ch04_structure_optimization/three_layer_timoshenko_beam_shape_optimization.py` | `three_layer_timoshenko_beam_shape_results` | Timoshenko beam network |
| 4 | `code/ch04_structure_optimization/arch_bridge_crawler_shape_control.py` | `results_arch_bridge_crawler_shape_control` | arch-bridge crawler |
| 6 Topological sensitivities and damage | `code/ch06_topological_sensitivity_damage/cantilever_truss_hybrid_minlp_pyomo_patched.py` | `results_cantilever_truss_hybrid_minlp` | hybrid MINLP truss design |
| 6 | `code/ch06_topological_sensitivity_damage/joint_recovery_switching_delta_d_adjoint.py` | `results_joint_recovery_switching_delta_d_adjoint` | joint recovery of switching and damage |
| 6 | `code/ch06_topological_sensitivity_damage/switching_tensegrity_crawler.py` | `results_switching_tensegrity_crawler` | switching tensegrity crawler |
| 10 Learning geometry and topological sensitivity | `code/ch10_learning_geometry_sensitivity/DDEC_top_der.py` | `results_ddec_scalar_poisson_sensitivity` | DDEC scalar Poisson sensitivity |
| 10 | `code/ch10_learning_geometry_sensitivity/bridge_truss_hante_insertion_growth_exact.py` | `results_bridge_hante_insertion_growth_exact` | greedy bar insertion, twin-arch bridge (truss) |
| 10 | `code/ch10_learning_geometry_sensitivity/bridge_frame_hante_insertion_growth_exact.py` | `results_bridge_frame_hante_insertion_growth_exact` | the same with Timoshenko frame members |
| 16 Timoshenko star graph with a hole | `code/ch16_timoshenko_star_hole/timoshenko_star_hole_total_energy.py` | `results_timoshenko_star_hole_energy` | Tables and Figures of Chapter 16 |

The computations of Chapter 18 are those of the doctoral dissertation of
M. Qian (Systems Research Institute, Polish Academy of Sciences, 2024);
the corresponding MATLAB code is available from its author.
<!-- TODO: add once permission is given (folder _pending_permission/). -->

Figures of Chapters 1, 5, 12 and 14 that are not listed here are schematic
drawings or were produced with earlier MATLAB codes that are not part of
this repository.
<!-- TODO (authors): scripts for the following figure folders were not found and
can be added if located: results_controlled_hybrid_graph_automaton,
results_controlled_hybrid_flow_branching_automaton (Ch. 5),
results_hybrid_damage_grid, results_bridge_damage_truss (Ch. 6),
results_bridge_damage_truss_asym, results_bridge_damage_truss_asym_load_only,
results_learning_compare (Ch. 11), Pictures/one-link-problems (Ch. 12, MATLAB). -->

## Requirements

Python 3.10 or newer and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

`torch` is needed only for scripts that train neural networks, `pyomo` and
`ortools` only for the MINLP example of Chapter 6 (which also needs a MINLP
solver usable from Pyomo, e.g. Bonmin or Couenne).

## Running

```
cd code/ch16_timoshenko_star_hole
python timoshenko_star_hole_total_energy.py
```

Results are written to the `results_*` folder next to the script.

## Citation

If you use this code, please cite the book (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
