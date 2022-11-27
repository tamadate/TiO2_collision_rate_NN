# TiO2_nanocluster_collision_rate
## Outline
* [1. Overview](#1-overview)
* [2. Requirements](#2-requirements)
* [3. Install](#3-install)
* [4. Usage](#4-usage)
* [5. Documentation](#5-documentation)
* [6. License](#6-license)
* [7. Reference](#7-reference)
## 1. Overview
This code is utilized to estimate gas phase TiO<sub>2</sub> nanoclusters (less than 3 nm) collision rate coeffcient (or collision kernel) via neural network (NN).  TiO<sub>2</sub> clusters are supposed to be electrically neutral.  The code is fully writen by Python3 with a NN library, PyTorch. The training data for the NN was collision probability mapping which generated from the molecular dynamics (MD) simulation with varying particle diameters, initial particle velocities, and collision parameter.  This type of collision rate coefficient calculation procedure with MD simulation is discussed in this paper [Goudeli et al., 2020](https://www.sciencedirect.com/science/article/pii/S0021850220300471?via%3Dihub) and some other papers as well.  A paper related to our TiO<sub>2</sub> simulation is under prepearation and it will be avairable more details, e.g., generation process of training data of MD simulation, detail of the NN, and validation of the NN in that paper.

## 2. Requirements
* [Anaconda](https://www.anaconda.com/)
* [Pytorch](https://pytorch.org/)
* If you don't use Anaconda, you need to have Python3 and Numpy.  Pytorch is required anyway.

## 3. Install
* Prepare the environment (see above 2. Requirements).
* Download or clone this repository.

## 4. Usage
Set parameters (temperature, 1st and 2nd clusters diameters) in `main.py` and run `main.py`, displaying the collision rate coefficienct, $\beta_{ij}$ and enhancement factor, $\eta_{ij}$ on your console.

## 5. Documentation
**Under construction...**

## 6. License
This code is an open-source package, meaning you can use or modify it under the terms and conditions of the GPL-v3 licence. You should have received a copy along with this package, if not please refer to [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).

## 7. Reference
**Under construction...**

# Author
* Dr. Tomoya Tamadate
* [Hogan Lab](https://hoganlab.umn.edu/)
* [LinkedIn](https://www.linkedin.com/in/tomoya-tamadate-953673142/)/[ResearchGate](https://www.researchgate.net/profile/Tomoya-Tamadate)/[Google Scholar](https://scholar.google.com/citations?user=XXSOgXwAAAAJ&hl=ja)
* University of Minnesota
* tamalab0109[at]gmail.com
