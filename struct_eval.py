import mixture
import random
import numarray

data = mixture.DataSet()

#data.fromFiles(["filt_WISC_WIAT_DISC_134.txt"]) # ,"DRD4_134_len.txt"
#m = mixture.readMixture("NEC_pheno_struct_11_nr5.mix")
#m = mixture.readMixture("NEC_pheno_struct_7nr3.mix")
#m = mixture.readMixture("NEC_pheno_struct_6nr4.mix")
#m = mixture.readMixture("NEC_pheno_struct_8nr3.mix")
#m = mixture.readMixture("NEC_pheno_struct_9nr3.mix")
#m = mixture.readMixture("NEC_pheno_struct_11_nr5.mix")

data.fromFiles(["filt_DAT_134.txt","DRD1_134.txt","DRD2_134.txt","DRD3_134.txt","DRD5_134.txt" ,"DRD4_134_len.txt"])

#m = mixture.readMixture("NEC_geno_14_struct2.mix")

#m = mixture.readMixture("NEC_geno_5_struct.mix")
#m = mixture.readMixture("NEC_geno_6_struct.mix")
m = mixture.readMixture("NEC_geno_7_struct_best.mix")
#m = mixture.readMixture("NEC_geno_8_struct3.mix")
#m = mixture.readMixture("NEC_geno_9_struct.mix")
#m = mixture.readMixture("geno_struct_50_7.mix")



#data.fromFiles(["filt_DAT_134.txt","DRD1_134.txt","DRD2_134.txt","DRD3_134.txt","DRD5_134.txt" ,"filt_WISC_WIAT_DISC_134.txt"])
#m = mixture.readMixture("NEC_ADHD_struct_7.mix")


data.internalInit(m)

c = m.classify(data,entropy_cutoff=0.50)
data.printClustering(m.G,c)
m.getClusterEntropy(data)
m.evalStructure(data.headers)

print m.groups

plot = numarray.zeros(( m.G,m.dist_nr ) )
for i in range(m.dist_nr):
    # check for noise variables
    if len(m.leaders[i]) == 1:
        l = m.leaders[i][0]
        for g in range(m.G):
           plot[g,i] = 1
    
    else:
        for l in m.leaders[i]:    
            
            if len(m.groups[i][l]) == 0:
                plot[l,i] = 2
            else:
               plot[l,i] = l+3
               for g in m.groups[i][l]:
                  plot[g,i] = l+3
print
for j in range(m.dist_nr):
    t = {}
    t[2] = 2
    t[1] = 1
        
    index = 3
    v_list = []
    for v in plot[:,j]:
        if v == 2:
            continue
        elif v == 1:
            break
        else:        
            if v not in v_list:
                v_list.append(v)

    v_list.sort()
    #print plot[:,j]
    #print v_list
    for v in v_list:
        t[v] = index
        index +=1
    
    for i in range(m.G):
        plot[i,j] = t[plot[i,j]]

space = 15
for i in range(len(plot[0])):
    slen = space - len(data.headers[i])
    print " "* slen + data.headers[i],
for i in range(len(plot)):
    print
    for j in range(len(plot[i])):
        print " "*(space-1) + str(plot[i,j]),
print "\n"
        
        

print "\n[",
for p in plot:
    print p.tolist(),";"
print "];\n\n"

for i in range(20):
    cut = i * 0.05
    z = m.classify(data,entropy_cutoff=cut,silent =1)
    t = numarray.where(z== -1)
    ind = t[0]
    print cut,":",len(z[ind])

m.printTraceback(data,c)