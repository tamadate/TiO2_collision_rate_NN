import math
import numpy as np
import os
import random
import sys

TotalLOOP=1.0
mass=np.array((47.867,16.00))
n1=9
n2=129
Y0=100
barray=np.arange(0.0,80.0,2.0)
Vini=float(sys.argv[1])


with open(str(n1)+".in","r") as f:
	lines=f.readlines()
	cluster=np.zeros((0,9))
	for line in lines:
		cluster=np.append(cluster,[line.split()],axis=0)

cluster1=cluster.astype(float)

with open(str(n2)+".in","r") as f:
	lines=f.readlines()
	cluster=np.zeros((0,9))
	for line in lines:
		cluster=np.append(cluster,[line.split()],axis=0)

cluster2=cluster.astype(float)

###################  Set center  ################
center=np.array((0.0,0.0,0.0))
V=np.array((0.0,0.0,0.0))
totalMass=0
for i in cluster1:
	center+=i[3:6]*mass[int(i[1])-1]
	V+=i[6:9]*mass[int(i[1])-1]
	totalMass+=mass[int(i[1])-1]

center/=totalMass
V/=totalMass

for i in cluster1:
	i[3:6]-=center
	i[6:9]-=V


Ncluster1=np.size(cluster1.T[0])

center=np.array((0.0,0.0,0.0))
V=np.array((0.0,0.0,0.0))
totalMass=0
for i in cluster2:
	center+=i[3:6]*mass[int(i[1])-1]
	V+=i[6:9]*mass[int(i[1])-1]
	totalMass+=mass[int(i[1])-1]

center/=totalMass
V/=totalMass

for i in cluster2:
	i[3:6]-=center
	i[6:9]-=V


Ncluster2=np.size(cluster2.T[0])

with open("TiO2","r") as f:
	lines_infile=f.readlines()

with open("TiO2","w") as f:
	flag=0
	for line in lines_infile:
		if(flag==0):
			f.write(line)
		if(line==" Atoms\n"):
			flag=1
	f.write("\n")
	for i in cluster1:
		f.write(str(int(i[0]))+"\t"+str(int(i[1]))+"\t"+str(i[2])+"\t"+str(i[3])+"\t"+str(i[4])+"\t"+str(i[5])+"\n")
	for i in cluster2:
		f.write(str(int(i[0]+Ncluster1))+"\t"+str(int(i[1]))+"\t"+str(i[2])+"\t"+str(i[3])+"\t"+str(i[4]+Y0)+"\t"+str(i[5])+"\n")


os.system("./lmp_mpi -in TiO2set.in")

with open("interData.dump","r") as f:
	linesID=f.readlines()
	loop=1
	for line in linesID:
		if(line=="ITEM: ATOMS id type q x y z vx vy vz\n"):
			I=loop
		loop+=1
	loop=0


for b in barray:
	ncollision=0
	for LOOP in np.arange(int(TotalLOOP)):

		with open("interDatain.dump","w") as f:
			for line in linesID[:I]:
				f.write(line)

			A=random.uniform(0,math.pi*2)
			B=random.uniform(0,math.pi*2)
			C=random.uniform(0,math.pi*2)
			A2=0#random.uniform(0,math.pi*2)
			B2=0#random.uniform(0,math.pi*2)
			C2=0#random.uniform(0,math.pi*2)
			VY0=-1*Vini

			for i in cluster1:
				X=np.cos(C)*(i[3]*np.cos(B)+i[4]*np.sin(A)*np.sin(B)-i[5]*np.cos(A)*np.sin(B))+np.sin(C)*(i[4]*np.cos(A)+i[5]*np.sin(A))
				Y=-np.sin(C)*(i[3]*np.cos(B)+i[4]*np.sin(A)*np.sin(B)-i[5]*np.cos(A)*np.sin(B))+np.cos(C)*(i[4]*np.cos(A)+i[5]*np.sin(A));
				Z=i[3]*np.sin(B)-i[4]*np.sin(A)*np.cos(B)+i[5]*np.cos(A)*np.cos(B)
				VX=np.cos(C)*(i[6]*np.cos(B)+i[7]*np.sin(A)*np.sin(B)-i[8]*np.cos(A)*np.sin(B))+np.sin(C)*(i[7]*np.cos(A)+i[8]*np.sin(A))
				VY=-np.sin(C)*(i[6]*np.cos(B)+i[7]*np.sin(A)*np.sin(B)-i[8]*np.cos(A)*np.sin(B))+np.cos(C)*(i[7]*np.cos(A)+i[8]*np.sin(A));
				VZ=i[6]*np.sin(B)-i[7]*np.sin(A)*np.cos(B)+i[8]*np.cos(A)*np.cos(B)
				f.write(str(int(i[0]))+"\t"+str(int(i[1]))+"\t"+str(i[2])+"\t"+str(X)+"\t"+str(Y)+"\t"+str(Z)+"\t"+str(VX)+"\t"+str(VY)+"\t"+str(VZ)+"\n")
			for i in cluster2:
				X=np.cos(C2)*(i[3]*np.cos(B2)+i[4]*np.sin(A2)*np.sin(B)-i[5]*np.cos(A2)*np.sin(B2))+np.sin(C2)*(i[4]*np.cos(A2)+i[5]*np.sin(A2))
				Y=-np.sin(C2)*(i[3]*np.cos(B2)+i[4]*np.sin(A2)*np.sin(B2)-i[5]*np.cos(A2)*np.sin(B2))+np.cos(C2)*(i[4]*np.cos(A2)+i[5]*np.sin(A2));
				Z=i[3]*np.sin(B2)-i[4]*np.sin(A2)*np.cos(B2)+i[5]*np.cos(A2)*np.cos(B2)
				VX=np.cos(C2)*(i[6]*np.cos(B2)+i[7]*np.sin(A2)*np.sin(B2)-i[8]*np.cos(A2)*np.sin(B2))+np.sin(C2)*(i[7]*np.cos(A2)+i[8]*np.sin(A2))
				VY=-np.sin(C2)*(i[6]*np.cos(B2)+i[7]*np.sin(A2)*np.sin(B2)-i[8]*np.cos(A2)*np.sin(B2))+np.cos(C2)*(i[7]*np.cos(A2)+i[8]*np.sin(A2));
				VZ=i[6]*np.sin(B2)-i[7]*np.sin(A2)*np.cos(B2)+i[8]*np.cos(A2)*np.cos(B2)
				f.write(str(int(i[0]+Ncluster1))+"\t"+str(int(i[1]))+"\t"+str(i[2])+"\t"+str(X+b)+"\t"+str(Y+Y0)+"\t"+str(Z)+"\t"+str(VX)+"\t"+str(VY+VY0)+"\t"+str(VZ)+"\n")


		os.system("./lmp_mpi -in TiO2.in")

		with open("takeover.dat","r") as f:
			lines=f.readlines()
		if lines==["1\n"]:
			LOOP-=1
		else:
			with open("result.dat","r") as f:
				lines=f.readlines()
			ncollision+=int(lines[0])

	with open("probability.dat","a") as f:
		f.write(str(b)+"\t"+str(Vini)+"\t"+str(float(ncollision)/TotalLOOP)+"\n")








