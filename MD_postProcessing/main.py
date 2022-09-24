import TiO2
import plot

## --------------   From MD simulation   -------------- ##
MDdata=TiO2.TiO2()
MDdata.sizeSet(3)       # 1:1nm, 2:2nm
MDdata.tempSet(1000)
MDdata.mapping()

plot=plot.plot()

## --------------   Analysis   -------------- ##
plot.probabilityMap(MDdata)
plot.flagMap(MDdata)
#plot.bindingLengthMap(MDdata,300)
#plot.BLmappingTemp(MDdata)
#plot.betaTemp(MDdata,200,1600)
#plot.enhanceTemp(MDdata,200,1600)
