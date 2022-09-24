
for v in 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0
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
