import TiO2
import plot

#"0.6nm","0.8nm","1nm","1.2nm","1.4nm","1.6nm","1.8nm","2nm","2.2nm","2.4nm","2.6nm","2.8nm","3nm"
## --------------   From MD simulation   -------------- ##
MDdata=TiO2.TiO2()
MDdata.sizeSet(2)       # 1:1nm, 2:2nm, 3:0.6nm, 4:3nm
MDdata.tempSet(1000)
MDdata.mapping()

plot=plot.plot()

MDdata.critical()

## --------------   Analysis   -------------- ##
#plot.probabilityMap(MDdata)
#plot.flagMap(MDdata)
#plot.bindingLengthMap(MDdata,300)
plot.BLmappingTemp(MDdata,6)
#plot.betaTemp(MDdata,200,1600)
#plot.enhanceTemp(MDdata,200,1600)
