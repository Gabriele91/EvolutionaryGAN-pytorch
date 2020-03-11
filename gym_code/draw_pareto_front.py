import sys
import matplotlib.pyplot as plt 

def readtsv(path):
    points = []
    with open(path,"r") as tsv:
        for line in tsv.readlines():
            point = line.split("\t")
            points.append((float(point[0]),float(point[1])))
    points.sort(key=lambda point: point[0]) #sort by x
    vx = [x for x,y in points]
    vy = [y for x,y in points]
    return vx,vy

def main(path,savepath):
    x,y = readtsv(path)
    if savepath == None:
        plt.plot(x,y)
        plt.show()
    else:
        fig = plt.figure()
        plt.plot(x,y)
        fig.savefig(savepath)



if __name__ == "__main__":
    if len(sys.argv) <= 1:
        exit(-1)
    elif len(sys.argv) == 2:
        main(sys.argv[1], None)
    elif len(sys.argv) > 2:
        main(sys.argv[1],sys.argv[2])