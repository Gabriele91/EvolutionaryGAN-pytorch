from dataset.datasets import toy_dataset
import h5py
from models import models_uncond
from lib.rng import py_rng, np_rng, t_rng
from lib.theano_utils import floatX, sharedX
from lib.data_utils import processing_img, convert_img_back, convert_img, Batch, shuffle, iter_data, ImgRescale, OneHot
from PIL import Image
from time import time
import shutil
import lasagne
import json
import theano.tensor as T
import theano
import numpy as np
import os
from matplotlib.pyplot import imshow, imsave, imread
import matplotlib.pyplot as plt
import matplotlib
import sys
import math
from sklearn.metrics import pairwise_kernels, pairwise_distances
import argparse
sys.path.append('..')
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
from matplotlib.colors import LogNorm
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def create_G(DIM=64):
    noise = T.matrix('noise')
    generator = models_uncond.build_generator_toy(noise, nd=DIM)
    Tgimgs = lasagne.layers.get_output(generator)
    gen_fn = theano.function([noise],lasagne.layers.get_output(generator, deterministic=True))
    return gen_fn, generator


def gen_color_map(X):
    # get the mesh
    m1, m2 = X[:, 0], X[:, 1]
    xmin = m1.min()-0.5
    xmax = m1.max()+0.5
    ymin = m2.min()-0.5
    ymax = m2.max()+0.5

    # get the density estimation 
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    kernel.set_bandwidth(bw_method=kernel.factor/2.5)
    Z = np.reshape(kernel(positions).T, X.shape) 
    plt.imshow(np.rot90(Z), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    

def generate_image(dist, num=0, desc=None, path=""):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    plt.clf()
    plt.margins(0,0)
    gen_color_map(dist)
    plt.savefig(path  + '.png', bbox_inches = 'tight', pad_inches = 0, dpi = 300)


def main(genpath,datasetname,outpath,target=False):
    #params
    DIM = 512
    SAMPLES = 3000 #3000
    nz = 2
    if target:
        #load samples from db
        xmb = toy_dataset(DATASET=datasetname, size=SAMPLES)
        generate_image(xmb, path=outpath)
    else:
        #load
        gen_fn, generator = create_G(DIM = DIM)
        #for all in the path:
        params_map = dict(np.load(genpath))
        params=list()
        for key,vals in sorted(params_map.items(),key=lambda x: int(x[0].split("_")[1])):
            params.append(np.float32(vals))
        #set params
        lasagne.layers.set_all_param_values(generator, params)
        # generate sample
        s_zmb = floatX(np_rng.uniform(-1., 1., size=(SAMPLES, nz)))
        g_imgs = gen_fn(s_zmb)
        generate_image(g_imgs, path=outpath)



if __name__ == '__main__':
    main(sys.argv[1], 
         sys.argv[2],
         sys.argv[3], 
         sys.argv[4]=="true" if len(sys.argv) >= 5 else False)