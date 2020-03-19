import sys
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

def main(path,savepath,invy=False):
    x,y,z = readtsv(path)
    if z is not None:
        print("MMD info: ", str(np.min(z))+" BEST,", str(np.average(z))+" +/- "+str(np.var(z))+" AVG")
    if invy:
        y_ = [-v for v in y]
        y = y_
    if savepath == None:
        plt.plot(x,y)
        plt.show()
        if z is not None:
            fig = plt.figure()
            z = np.array(z)
            z = 1.-(np.array(z)-np.min(z)) / (np.max(z)-np.min(z)) 
            plt.plot(x,y)
            plt.xlabel('Quality')
            plt.ylabel('Diversity')
            plt.scatter(x, y, c=z, s=(z*13.0) ** 2, marker='o')
            plt.show()
    else:
        fig = plt.figure()
        plt.plot(x,y)
        fig.savefig(savepath)
        if z is not None:
            pathsplit = savepath.split('.')
            pathfile, ext = "".join(pathsplit[:-1]), pathsplit[-1] 
            fig = plt.figure()
            z = np.array(z)
            z = 1.-(np.array(z)-np.min(z)) / (np.max(z)-np.min(z)) 
            plt.plot(x,y)
            plt.xlabel('Quality')
            plt.ylabel('Diversity')
            plt.scatter(x, y, c=z, s=(z*13.0) ** 2, marker='o')
            cbar = plt.colorbar()
            cbar.set_label('MMD normalized')
            fig.savefig(savepath+"_mmd."+ext)



if __name__ == "__main__":
    if len(sys.argv) <= 1:
        exit(-1)
    elif len(sys.argv) == 2:
        main(sys.argv[1], None)
    elif len(sys.argv) > 2:
        invy = sys.argv[3].lower() == "true" if len(sys.argv) > 3  else False
        main(sys.argv[1],sys.argv[2],invy)