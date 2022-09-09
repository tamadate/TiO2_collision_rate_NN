# TiO2_nanocluster_collision_rate
## Overview
This code is utilized to estimate gas phase TiO2 nanoclusters (less than 3nm) collision rate coeffcient (or collision kernel) at arbitrary temperature using neural network (NN).  The code is fully writen by Python3 which PyTorch is used as a NN library. The training data of this NN was generated from the molecular dynamics (MD) simulation with varying particle diameters, initial particle velocities, and collision parameter.  This type of collision rate coefficient calculation procedure with MD simulation is discussed in this paper [Goudeli et al., 2020](https://www.sciencedirect.com/science/article/pii/S0021850220300471?via%3Dihub) and some other papers as well.  Our paper related to this simulation is under prepearation and it will be avairable the generation process of training data with MD simulation, detail of the neural network, and validation of the neural nerwork in that paper.
## Usage
### 1. Collsion probability mapping calculation
A script `maiiping.py` is used for this collision probability calculation mode and it estimates the collision probability of two nanoclusters from three inputs: reaction temperature and diameters of clusters. As a example, calculation result of 1.0 nm - 1.0 nm particles collision at 300 K is shown in Figure 1.
### 2. Collsion kernel mapping calculation
# Author
* Dr. Tomoya Tamadate
* [LinkedIn](https://www.linkedin.com/in/tomoya-tamadate-953673142/)/[ResearchGate](https://www.researchgate.net/profile/Tomoya-Tamadate)/[Google Scholar](https://scholar.google.com/citations?user=XXSOgXwAAAAJ&hl=ja)
* University of Minnesota ([Hogan Lab](https://hoganlab.umn.edu/))
* tamalab0109[at]gmail.com
