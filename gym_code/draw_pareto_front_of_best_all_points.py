import sys
import os
import numpy as np
import matplotlib.pyplot as plt 

def readtsv(path):
    metric = False
    points = []
    with open(path,"r") as tsv:
        for line in tsv.readlines():
            point = line.split("\t")
            if len(point) == 3:
                metric = True
            if metric:
                points.append((float(point[0]),float(point[1]), float(point[2])))
            else:
                points.append((float(point[0]),float(point[1])))
    points.sort(key=lambda point: point[0]) #sort by x
    vx = [p[0] for p in points]
    vy = [p[1] for p in points]
    vz = None
    if metric:
        vz = [abs(p[2]) for p in points]
    return vx,vy,vz

def main(path,savepath,invy=False,invx=False, cut=0):
    minname = []
    x,y,z = [],[],[]
    for root, dirs, files in os.walk(path):
        files.sort(key=lambda x: int(x.split("_")[0]))
        for filename in files:
            if filename.rfind(".tsv") >= 0:
                idfile = filename.split("_")[0]
                if int(idfile) >= 0:
                    currfilepath = os.path.join(root, filename)
                    curr_x_y_z = readtsv(currfilepath)
                    min_z = min(curr_x_y_z[2])
                    x.extend(curr_x_y_z[0])
                    y.extend(curr_x_y_z[1])
                    z.extend(curr_x_y_z[2])
                    minname.append((min_z,currfilepath))

    if len(minname):
        minname.sort(key=lambda z_name: z_name[0])
        for zfit,name in minname[:10]:
            print(name, zfit)
    
    x,y,z = zip(*sorted(zip(x, y, z), key=lambda xyz: -xyz[2]))

    if z is not None:
        print("MMD info: ", str(np.min(z))+" BEST,", str(np.average(z))+" +/- "+str(np.var(z))+" AVG")
    if invy:
        y_ = [-v for v in y]
        y = y_
    if invx:
        x_ = [-v for v in x]
        x = x_
    if savepath == None:
        #plt.plot(x,y)
        #plt.show()
        if z is not None:
            fig = plt.figure()
            z = np.array(z)
            #z = 1.-(np.array(z)-np.min(z)) / (np.max(z)-np.min(z)) 
            #plt.plot(x,y)
            plt.xlabel('Quality')
            plt.ylabel('Diversity')
            plt.scatter(x, y, c=z, s=(z*13.0) ** 2.1, marker='o')
            plt.show()
    else:
        fig = plt.figure()
        if z is not None:
            pathsplit = savepath.split('.')
            pathfile, ext = "".join(pathsplit[:-1]), pathsplit[-1] 
            fig = plt.figure()
            z = np.array(z)
            rnk = (z.argsort()).argsort() 
            rnk_norm = 1.0-(np.array(rnk)-np.min(rnk)) / (np.max(rnk)-np.min(rnk)) 

            #x,y,z,rnk,rnk_norm = zip(*sorted(zip(
            #    x, 
            #    y, 
            #    z.tolist(), 
            #    rnk.tolist(), 
            #    rnk_norm.tolist() ), 
            #    key=lambda a: a[4])
            #)
            if cut>0:
                x = x[len(x)-cut:]
                y = y[len(y)-cut:]
                z = z[len(z)-cut:]
                rnk = rnk[len(rnk)-cut:]
                rnk_norm = rnk_norm[len(rnk_norm)-cut:]

            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            rnk = np.array(rnk)
            rnk_norm = np.array(rnk_norm)
                            
            factor=np.power(100,rnk_norm*2)
            factor_norm = (np.array(factor)-np.min(factor)) / (np.max(factor)-np.min(factor)) 

            print(len(x),len(y),len(z),len(rnk),len(rnk_norm), len(factor_norm))

            plt.xlabel('Quality')
            plt.ylabel('Diversity')
            plt.scatter(x, y, c=factor_norm, s=np.power(100,factor_norm),marker='o')
            cbar = plt.colorbar()
            cbar.set_label('100^rank(MMD)')
            if cut>0:
                fig.savefig(savepath+"_"+str(cut)+"."+ext)
            else:
                fig.savefig(savepath+"."+ext)




if __name__ == "__main__":
    if len(sys.argv) <= 1:
        exit(-1)
    elif len(sys.argv) == 2:
        main(sys.argv[1], None)
    elif len(sys.argv) == 3:
        main(sys.argv[1],sys.argv[2],invy)
    elif len(sys.argv) == 4:
        invy = sys.argv[3].lower() == "true"
        main(sys.argv[1],sys.argv[2],invy)
    elif len(sys.argv) == 5:
        invy = sys.argv[3].lower() == "true"
        invx = sys.argv[4].lower() == "true"
        main(sys.argv[1],sys.argv[2],invy, invx)
    elif len(sys.argv) > 5:
        invy = sys.argv[3].lower() == "true"
        invx = sys.argv[4].lower() == "true"
        cut = int(sys.argv[5])
        main(sys.argv[1],sys.argv[2],invy, invx, cut)