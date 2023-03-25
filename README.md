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
This code is designed to estimate the collision (or coagulation) rate coefficient of gas-phase TiO<sub>2</sub> nanoclusters with a diameter less than 3 nm and electrically neutral state, using a neural network implemented in Python 3 with the PyTorch library. The neural network model has been trained using data generated from molecular dynamics (MD) simulations that account for both Van der Waals and dipole interactions. As a result, the trained neural network model is expected to accurately reproduce the collision kinetics at a molecular level, with variation in particle diameters, initial particle velocities, and collision parameters taken into account.

## 2. Requirements
* [Anaconda](https://www.anaconda.com/)
* [Pytorch](https://pytorch.org/)
* If you don't use Anaconda, you need to have Python3 and Numpy.  Pytorch is required anyway.

## 3. Installation
* Prepare the environment (see above 2. Requirements).
* Download or clone this repository.

## 4. Usage
The directories `NN_training` and `NN_learned` contain codes for the neural network training process and the trained neural network model, respectively. To obtain the collision rate coefficient, $\beta_{ij}$, and the enhancement factor, $\eta_{ij}$, run the trained neural network model `main.py` in the `NN_learned` directory with the desired calculation parameters (temperature, diameter of the first and second clusters) specified. The detailed instructions can be found in the documentation. The results will be displayed on the console.

## 5. Documentation
**Under construction....**

## 6. License
This code is an open-source package, meaning you can use or modify it under the terms and conditions of the GPL-v3 licence. You should have received a copy along with this package, if not please refer to [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).

## 7. Reference
[Tamadate, T., Yang, S., & Hogan, C. J., Jr. (2023). A neural network parametrized coagulation rate model for <3 nm titanium dioxide nanoclusters. Journal of Chemical Physics, 158(8)](https://aip.scitation.org/doi/abs/10.1063/5.0136592)

# Author
* Dr. Tomoya Tamadate
* [Hogan Lab](https://hoganlab.umn.edu/)
* [LinkedIn](https://www.linkedin.com/in/tomoya-tamadate-953673142/)/[ResearchGate](https://www.researchgate.net/profile/Tomoya-Tamadate)/[Google Scholar](https://scholar.google.com/citations?user=XXSOgXwAAAAJ&hl=ja)
* University of Minnesota
* tamad005[at]umn.edu
