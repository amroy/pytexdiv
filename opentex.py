#!/usr/bin/python3

import os, glob
import math
import argparse
import numpy as np
import pywt
from PIL import Image


class Model(object):
    def __init__(self, name):
        """ The main class of the statistical model """
        self.name = name

    def fit(self, data):
        pass


def main():
    print("************************************************************")
    print("Welcome to the statistical texture content retrieval program")
    print("       Developed and maintained by Hassan Rami")
    print("             hassan.rami@outlook.com")
    print("                   Version 1.0")
    print("           Last update: September 2019")
    print("************************************************************")

    """ Read command line arguments """
    parser = argparse.ArgumentParser(description="Statistical analysis of textured images")
    parser.add_argument('-ds', metavar='dataset', help='Dataset path')
    parser.add_argument('-m', metavar='model', help='Model type (ggd, weibull, mog, mogg, mowbl...)')
    parser.add_argument('-t', metavar='transform', help='Transform type (dwt, udwt, dtcwt, ct ...)')
    parser.add_argument('--scales', default=3, metavar='scales', help='Number of scales (default is 3)')
    parser.add_argument('--mag', default=True, metavar='magnitude', help='Model magnitude in case of complex coefficients')
    parser.add_argument('--phase', default=False, metavar='phase', help='Model phase information in case of complex coefficients ')
    parser.add_argument('-d', '--div', metavar='divergence_type', default='kld', help='Divergence type (kld or csd)')
    parser.add_argument('-gpu', default=True, help='Use accelerated computation using GPU programming')
    parser.add_argument('-mci', default=False, help='Use Monte-Carlo integration for non-analytic divergences')
    parser.add_argument('-s', '--save', default=True, help='Save results in a text file')
    args = parser.parse_args()

    """ Perform the wavelet transform and extract features from each images """
    model = Model(args.m)
    images_list = glob.glob(args.ds)
    if len(images_list) == 0:
        raise IOError("Dataset path empty!")
    im_filename, img_ext = os.path.splitext(images_list[0])
    dataset_size = 0
    for i in range(dataset_size):
        image = Image.open(images_list[i])
        """ Apply the transform """
        img_features = np.array(args.scales * 3, dtype=float)
        for lvl in range(args.scales):
            LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
            features = model.fit(LH)


if __name__ == "__main__":
    main()
