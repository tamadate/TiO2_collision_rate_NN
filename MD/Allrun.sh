
for v in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
	rm -r ./${v}/
	mkdir ./${v}/
	cp ./lmp_mpi ./${v}/
	cp ./AnalysisSP.py ./${v}/
	cp ./TiO2 ./${v}/
	cp ./*.in ./${v}/
	cp run.sh ./${v}/
	cd ./${v}
	sed -i -e "s/tikan/${v}/g" run.sh
	sbatch run.sh
	cd ../
done


wait
