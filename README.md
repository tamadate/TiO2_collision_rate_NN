# TiO2_nanocluster_collision_rate
## Outline
* [1. Overview](#1-overview)
* [2. Requirements](#2-requirements)
* [3. Installation](#3-installation)
* [4. Usage](#4-usage)
* [5. Documentation](#5-documentation)
* [6. License](#6-license)
* [7. Reference](#7-reference)
## 1. Overview
This code is utilized to estimate gas phase TiO<sub>2</sub> nanoclusters (less than 3 nm and electrially neutral) collision (or coagulation) rate coeffcient via neural network.  The code is fully writen by Python3 with a neural network library, PyTorch. The training data for the neural network was generated from the molecular dynamics (MD) simulation including Van der Waals and dipole interactions, meaning the trained neural network model is expected reproduce the collision kinetics in molecular level. varying particle diameters, initial particle velocities, and collision parameter.

## 2. Requirements
* [Anaconda](https://www.anaconda.com/)
* [Pytorch](https://pytorch.org/)
* If you don't use Anaconda, you need to have Python3 and Numpy.  Pytorch is required anyway.

## 3. Installation
* Prepare the environment (see above 2. Requirements).
* Download or clone this repository.

## 4. Usage
Two directories, `NN_training` and `NN_learned` respectively store the codes for neural network training process and trained neural network model.  Running trained neural network model `main.py` in `NN_learned` directory with arbitrary calculation parameters (temperature, 1st and 2nd clusters diameters) return the collision rate coefficient, $\beta_{ij}$ and enhancement factor, $\eta_{ij}$ on your console.  The detail usage is shown in documentation.

## 5. Documentation
**Under construction....**

## 6. License
This code is an open-source package, meaning you can use or modify it under the terms and conditions of the GPL-v3 licence. You should have received a copy along with this package, if not please refer to [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).

## 7. Reference
[Tamadate, T., Yang, S., & Hogan, C. J., Jr. (2023). A neural network parametrized coagulation rate model for <3 nm titanium dioxide nanoclusters. Journal of Chemical Physics, 158(8) doi:10.1063/5.0136592](https://aip.scitation.org/doi/abs/10.1063/5.0136592)

# Author
* Dr. Tomoya Tamadate
* [Hogan Lab](https://hoganlab.umn.edu/)
* [LinkedIn](https://www.linkedin.com/in/tomoya-tamadate-953673142/)/[ResearchGate](https://www.researchgate.net/profile/Tomoya-Tamadate)/[Google Scholar](https://scholar.google.com/citations?user=XXSOgXwAAAAJ&hl=ja)
* University of Minnesota
* tamad005[at]umn.edu
