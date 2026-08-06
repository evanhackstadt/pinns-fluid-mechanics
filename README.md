# pinns-fluid-mechanics
Research performed at Oregon Health & Science University as part of the ORION Internship.
Principal Investigator: Dr. Sandra Rugonyi

## Description
Proof-of-concept work on Physics-Informed Neural Networks (PINNs) for solving patient-specific hemodynamics in the context obstructive cardiovascular conditions. The project builds up in complexity through multiple scenarios. All problems consider a simplified problem: steady-state laminar flow through a 2D channel, with the stenosis obstruction in scenarios 2-5 modeled by half an ellipse.

## Repository Overview
* `scenarios/` contains distinct problem setups or goals. See the READMEs in scenarios 2-5.
  * `1-poiseuille-flow`: sanity check to get the PINN working; no obstruction
  * `2-stenosis`: add ellipse obstruction
  * `3-stenosis-inlet-u`: change boundary conditions; bug fixes
  * `4-supervised`: add semi-supervised training; perform error analysis on `n_labeled_points`
  * `5-geometry-conditioned`: train across multiple geometries; assess generalization

## Citations
All PINN models were implemented using the DeepXDE Python library:
```
@article{lu2021deepxde,
  author  = {Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
  title   = {{DeepXDE}: A deep learning library for solving differential equations},
  journal = {SIAM Review},
  volume  = {63},
  number  = {1},
  pages   = {208-228},
  year    = {2021},
  doi     = {10.1137/19M1274067}
}
```
The FEM solver was implemented using the FEniCS Python library:
```
@misc{BarattaEtal2023,
  title     = {{DOLFINx}: the next generation {FEniCS} problem solving environment},
  author    = {Baratta, Igor A. and Dean, Joseph P. and Dokken, J{\o}rgen S. and Habera, Michal and Hale, Jack S. and Richardson, Chris N. and Rognes, Marie E. and Scroggs, Matthew W. and Sime, Nathan and Wells, Garth N.},
  doi       = {10.5281/zenodo.10447666},
  year      = {2023},
  howpublished = {preprint}
}
```
```
@article{AlnaesEtal2014,
  title     = {Unified Form Language: A domain-specific language for weak formulations of partial differential equations},
  author    = {Alnaes, Martin S. and Logg, Anders and {\O}lgaard, Kristian B. and Rognes, Marie E. and Wells, Garth N.},
  journal   = {{ACM} Transactions on Mathematical Software},
  year      = {2014},
  volume    = {40},
  doi       = {10.1145/2566630},
}
```
