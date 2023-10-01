import numpy as np
import os
import matplotlib.pyplot as plt
folder = os.path.split(os.path.realpath(__file__))[0]


def read_data(f, n):

    ret = []
    file = open(f+"/processor" + str(0)+"/ISAT_VLE/ISAT_VLE/TreeSize", 'r')
    data_t = [s.split(",") for s in file.read().split("\n")[:-1]]
    data_t = np.array([[float(s2) for s2 in s1] for s1 in data_t])
    ret.append(data_t[:,0])
    
    for i in range(n):
        file = open(f+"/processor" + str(i)+"/ISAT_VLE/ISAT_VLE/TreeSize", 'r')
        data_t = [s.split(",") for s in file.read().split("\n")[:-1]]
        data_t = np.array([[float(s2) for s2 in s1] for s1 in data_t])
        ret.append(data_t[:,1])
    ret = np.array(ret) 
    return ret
    return np.array([data[:, 0], data[:, 1], data2[:, 1]])

def read_data2(f, n):

    ret = []
    file = open(f+"/processor" + str(0)+"/ISAT_VLE/ISAT_VLE/TreeSize", 'r')
    data_t = [s.split(",") for s in file.read().split("\n")[:-1]]
    data_t = np.array([[float(s2) for s2 in s1] for s1 in data_t])
    ret.append(data_t[:,0])

    file = open(f+"/processor" + str(0)+"/ISAT_VLE/ISAT_VLE/TreeSize", 'r')
    data_t = [s.split(",") for s in file.read().split("\n")[:-1]]
    data_t = np.array([[float(s2) for s2 in s1] for s1 in data_t])
    ret.append(data_t[:,1])
    
    for i in range(n):
        file = open(f+"/processor" + str(i)+"/ISAT_VLE/ISAT_VLE/TreeSize", 'r')
        data_t = [s.split(",") for s in file.read().split("\n")[:-1]]
        data_t = np.array([[float(s2) for s2 in s1] for s1 in data_t])
        ret.append(data_t[:,2])
    ret = np.array(ret) 
    return ret


data = read_data(folder+"/CPU_data/lb_128_p2", 128)
data2 = read_data2(folder+"/CPU_data/lb3_128_LM", 128)
data3 = read_data2(folder+"/CPU_data/np_128_T", 128)
#data4 = read_data2(folder+"/CPU_data/2_lb3_long", 2)


 #----------------------------------------------
def moving_average(interval,windowsize):
    interval = np.concatenate((interval,np.ones(int(windowsize))*interval[-1]),axis=None)
    window = np.ones(int(windowsize))/float(windowsize)
    re = np.convolve(interval,window,'same')
    return re[:-windowsize]
#----------------------------------------------
'''
file = open(folder +"/ptime_data", 'w')
file.write("\n".join([",".join([str(s1) for s1 in s2]) for s2 in data.transpose()]))


plt.plot(data[0][0::10], moving_average(data[1],40)[0::10],  color= "b")
#plt.plot(data1[0][0::10], moving_average(data1[2]-data1[1],10)[0::10],  color= "r")

plt.savefig(folder + "//time_noISAT.png", dpi=400 , bbox_inches='tight') '''

print(np.sum(data[1:,0]))

data_sum= [np.sum(data[1:,i]) for i in range(len(data[1]))]
data2_sum= [np.sum(data2[1:,i]) for i in range(len(data2[1]))]
data3_sum= [np.sum(data3[1:,i]) for i in range(len(data3[1]))]
#print(data3_sum)
#plt.xlim(0,2e-6)
plt.xlim(0,1e-6*1e6)

#plt.plot(data[0][0::10]*1e6, moving_average(data[1],40)[0::10],  color= "r")
plt.plot(data3[0][0::10]*1e6, moving_average(data3_sum,40)[0::10]/1e6,  color= "b")
plt.plot(data[0][0::10]*1e6, moving_average(data[1],40)[0::10]/1e6,  color= "black")


plt.plot(data2[0][0::10]*1e6, moving_average(data2_sum,40)[0::10]/1e6,  color= "g")

plt.plot(data2[0][0::10]*1e6, moving_average(data2[1],40)[0::10]/1e6,  color= "g")
#plt.plot(data4[0][0::10]*1e6, moving_average(data4[1]+data4[2]+data4[4],40)[0::10],  color= "g")

#plt.plot(data2[0][0::10]*1e6, moving_average(data2[2],40)[0::10],  color= "b")
#plt.plot(data2[0][0::10]*1e6, moving_average(data2[1],40)[0::10],  color= "b")



#plt.ylim(0,3.5)
plt.xlabel(r"physical time ($\mu$s)", fontsize=18)
plt.ylabel("Table size (million)", fontsize=18)
#plt.title("(a)",loc='left', fontsize=18)
plt.title("(b)",loc='left', fontsize=18)
#plt.yscale('log')
plt.yticks(fontsize=15)
plt.xticks(np.arange(0, 2.1, 0.5),fontsize=15)
#plt.xticks(np.arange(0, 2.1, 0.5),fontsize=15)
#plt.legend( ["Original ISAT","Shared ISAT with LB","Shared & local ISAT with LB"],loc="upper left",frameon=False,fontsize=15)
ax = plt.gca()
ax.xaxis.get_offset_text().set_fontsize(15)
plt.savefig(folder + "//mem_core_128.png", dpi=400 , bbox_inches='tight')


data2_sum = data2[1]+data2[2]

print(np.max(data2_sum))
print(np.max(data[1]))
print(np.max(data3[1]))
#print(np.max(data4[1]+data4[2]+data4[4]))