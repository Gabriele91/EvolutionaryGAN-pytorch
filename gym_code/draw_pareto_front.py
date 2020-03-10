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

def main(path):
    x,y = readtsv(path)
    plt.plot(x,y)
    plt.show()



if __name__ == "__main__":
    main(sys.argv[1])