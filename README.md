# TiO2_nanocluster_collision_rate
## 1. Overview
This code is utilized to estimate gas phase TiO<sub>2</sub> nanoclusters (less than 3 nm) collision rate coeffcient (or collision kernel) via neural network (NN).  TiO<sub>2</sub> clusters are supposed to be electrically neutral.  The code is fully writen by Python3 with a NN library, PyTorch. The training data for the NN was collision probability mapping which generated from the molecular dynamics (MD) simulation with varying particle diameters, initial particle velocities, and collision parameter.  This type of collision rate coefficient calculation procedure with MD simulation is discussed in this paper [Goudeli et al., 2020](https://www.sciencedirect.com/science/article/pii/S0021850220300471?via%3Dihub) and some other papers as well.  A paper related to our TiO<sub>2</sub> simulation is under prepearation and it will be avairable more details, e.g., generation process of training data of MD simulation, detail of the NN, and validation of the NN in that paper.

## 2. Usage
Download 

## 3. License
This code is an open-source package, meaning you can use or modify it under the terms and conditions of the GPL-v3 licence. You should have received a copy along with this package, if not please refer to [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).

# Author
* Dr. Tomoya Tamadate
* [LinkedIn](https://www.linkedin.com/in/tomoya-tamadate-953673142/)/[ResearchGate](https://www.researchgate.net/profile/Tomoya-Tamadate)/[Google Scholar](https://scholar.google.com/citations?user=XXSOgXwAAAAJ&hl=ja)
* University of Minnesota ([Hogan Lab](https://hoganlab.umn.edu/))
* tamalab0109[at]gmail.com
