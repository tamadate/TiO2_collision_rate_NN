# TiO2_nanocluster_collision_rate
## Overview
This code is utilized to estimate gas phase TiO<sub>2</sub> nanoclusters (less than 3 nm) collision rate coeffcient (or collision kernel) via neural network (NN).  TiO<sub>2</sub> clusters are supposed to be electrically neutral.  The code is fully writen by Python3 with PyTorch as a NN library. The training data for the NN was collision probability mapping which generated from the molecular dynamics (MD) simulation with varying particle diameters, initial particle velocities, and collision parameter.  This type of collision rate coefficient calculation procedure with MD simulation is discussed in this paper [Goudeli et al., 2020](https://www.sciencedirect.com/science/article/pii/S0021850220300471?via%3Dihub) and some other papers as well.  A paper related to our TiO<sub>2</sub> simulation is under prepearation and it will be avairable more details, e.g., generation process of training data of MD simulation, detail of the NN, and validation of the NN in that paper.
***
##Theory
### 1. Collsion probability mapping calculation
The two TiO<sub>2</sub> nanoclusters (*D*<sub>p,1</sub> and *D*<sub>p,2</sub> in diameters) collision probability is changed by impact parameter (*b*) and initial velocity (*v*<sub>0</sub>) as shown in Figure 1(a).  Generally, the collision probability is increased by decreasing *v*<sub>0</sub> because the Van der Waals potential is more subjected and also collision is promoted by decreaseing *b*.  A script maiiping.py is used to estimate that collision probability mapping, *p*(*b*,*v*<sub>0</sub>) from MD based NN model.  As an example, calculated collision probability mapping with *D*<sub>p,1</sub> = *D*<sub>p,2</sub> = 1.0 nm is shown in Figure 1(b).
### 2. Collsion kernel mapping calculation
$$
\beta_{ij}=\int_0^\infty{ \int_0^\infty 2 \pi b \cdot p(b,v_0) \cdot v_0f(v_0) db dv_0}   \\
f(v_0)=\left({m_{ij} \over{2 \pi k_b T}} \right)^{3/2} 4 \pi v_0^2 exp \left( -{m_{ij} v_0^2 \over {2 k_b T}} \right)
$$
***
## Usage



# Author
* Dr. Tomoya Tamadate
* [LinkedIn](https://www.linkedin.com/in/tomoya-tamadate-953673142/)/[ResearchGate](https://www.researchgate.net/profile/Tomoya-Tamadate)/[Google Scholar](https://scholar.google.com/citations?user=XXSOgXwAAAAJ&hl=ja)
* University of Minnesota ([Hogan Lab](https://hoganlab.umn.edu/))
* tamalab0109[at]gmail.com
