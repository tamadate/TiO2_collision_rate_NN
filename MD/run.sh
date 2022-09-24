#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --mem=20gb
#SBATCH --time 80:00:00

module load python
module load lammps

for i in 0 1 2 3 4 5 6 7 8 9
do
	mkdir ./${i}/
	cp ./lmp_mpi ./${i}/
	cp ./AnalysisSP.py ./${i}/
	cp ./TiO2 ./${i}/
	cp ./*.in ./${i}/
	cd ./${i}
	python AnalysisSP.py tikan &
	cd ../
done


wait
