#!/usr/bin/python3

import os
import numpy as np
import math
import time, datetime
import matplotlib.pyplot as plt
import argparse
import csv
import pyopencl as cl
import pyopencl.array as cl_array


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ModelSet(object):
    def __init__(self, filename, model, div, mci=False, gpu=False):
        self.filename = filename
        self.model = model
        self.div = div
        self.models = []
        self.mci = mci
        self.gpu = gpu
        self.nb_models = 0
        self.nb_features = 0

        if not os.path.exists(filename):
            raise Exception('Models file not found!')
        else:
            self.load()

        # Setup OpenCL environment
        self.cl_platform = cl.get_platforms()[0]
        try:
            device = self.cl_platform.get_devices()[1]
            print("The following GPU device will be used for fast parallel computation:")
            print(device.get_info(cl.device_info.NAME))
        except IndexError:
            print("GPU computation is not supported on this machine! CPU will be used instead.")
            device = self.cl_platform.get_devices()[0]

        self.cl_context = cl.Context([device])
        self.cl_queue = cl.CommandQueue(self.cl_context)

    def load(self):
        try:
            self.models = [line.split(',') for line in open(self.filename, 'r')]
            print(':: Models file loaded: {:}'.format(self.filename))
        except IOError:
            print("Could not read the models file!")

        self.nb_models = len(self.models)
        self.nb_features = len(self.models[0])

        for i in range(self.nb_models):
            for j in range(self.nb_features):
                self.models[i][j] = float(self.models[i][j])

    def kld(self):
        divs = []
        arr = 0
        return divs, arr

    def csd(self):
        divs = []
        arr = 0
        return divs, arr

    def evaluate(self):
        if not self.models:
            raise Exception('Models required for evaluation!')

        print("Computing divergences between {:} models with {:} features per model".format(self.nb_models, self.nb_features))

        if self.gpu:
            models = np.empty_like(self.models, dtype=np.float32)
            divs = np.ndarray((self.nb_models, self.nb_models), dtype=np.float32)

            for i in range(self.nb_models):
                for j in range(self.nb_features):
                    models[i][j] = float(self.models[i][j])

            models_buffer = cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=models)
            divs_buffer = cl.Buffer(self.cl_context, cl.mem_flags.WRITE_ONLY, divs.nbytes)

            cl_kernel_name = self.model + '_' + self.div
            cl_kernel_filename = "cl_kernels/" + cl_kernel_name + ".cl"
            if not os.path.exists(cl_kernel_filename):
                raise IOError("OpenCL Kernel for model {:} and divergence {:} not found in location cl_kernels".format(self.model, self.div))

            program = cl.Program(self.cl_context, open(cl_kernel_filename).read()).build()

            if self.div == 'kld':
                print(':: [{:}] Kullback-Leibler divergence...'.format(now()))
                program.ggd_kld(self.cl_queue, divs.shape, None, models_buffer, divs_buffer, np.int32(self.nb_models), np.int32(self.nb_features))
                cl.enqueue_copy(self.cl_queue, divs, divs_buffer).wait()
            elif self.div == 'csd':
                print(':: Cauchy-Schwarz divergence...')
            else:
                raise Exception('Divergence {:} not supported!'.format(self.div))
        else:
            if self.div == 'kld':
                print(':: [{:}] Kullback-Leibler divergence...'.format(now()))
                divs = self.kld()
            elif self.div == 'csd':
                print(':: Cauchy-Schwarz divergence...')
                divs = self.csd()
            else:
                raise Exception('Divergence {:} not supported!'.format(self.div))

        """ TODO: Compute the average retrieval rate """
        arr = 0
        NS = 16
        r = np.zeros((self.nb_models,), dtype=int)
        r_shape = r.shape
        rr = np.ndarray((NS, self.nb_models), dtype=int)
        """ Compute the average retrieval rate """
        for q in range(self.nb_models):
            si = np.argsort(divs[q])
            c = math.floor(q / NS)
            r[si] = range(self.nb_models)
            start = int(c*NS)
            end = int((c+1)*NS - 1)
            for i in range(NS):
                rr[i, q] = r[start+i]

        pr = np.empty((self.nb_models,))
        for q in range(self.nb_models):
            col = rr[:, q]
            a = [x for x in col if x < NS]
            pr[q] = len(a)

        arr = np.mean(pr) / NS * 100

        print(":: Average Retrieval Rate is {:}".format(arr))

        return divs, arr


class WeibullModelSet(ModelSet):
    def csd(self):
        return []

    def kld(self):
        C = 0.5772
        divs = np.ndarray((self.nb_models, self.nb_models), dtype=float)

        t_start = time.time()
        for i in range(self.nb_models):
            for j in range(self.nb_models):
                d = 0
                for k in range(0, self.nb_features, 2):
                    a1 = self.models[i][k]
                    b1 = self.models[i][k+1]
                    a2 = self.models[j][k]
                    b2 = self.models[j][k+1]
                    d += np.log(b1 / (math.pow(a1, b1))) - np.log(b2 / math.pow(a2, b2)) + (np.log(a1) - C/b1)*(b1-b2) + math.pow((a1/a2), b2) * math.gamma(b2/b1+1) - 1
                divs[i][j] = d
            if i == 3:
                print(":: [{:}] All divergences will complete in {:} seconds".format(now(), (time.time() - t_start)/3 * self.nb_models))

        return divs


class GGDModelSet(ModelSet):
    def csd(self):
        return []

    def kld(self):
        divs = np.ndarray((self.nb_models, self.nb_models), dtype=float)

        t_start = time.time()
        for i in range(self.nb_models):
            for j in range(self.nb_models):
                d = 0
                for k in range(0, self.nb_features, 2):
                    a1 = self.models[i][k]
                    b1 = self.models[i][k+1]
                    a2 = self.models[j][k]
                    b2 = self.models[j][k+1]
                    try:
                        d += math.pow(a1/a2, b2) * (math.gamma((b2+1) / b1) / math.gamma(1/b1)) - 1 / b1 + math.pow(a2/a1, b1) * (math.gamma((b1+1) / b2) / math.gamma(1/b2)) - 1/b2
                    except OverflowError:
                        d += 0
                        print("OverflowError in KLD GGD")
                divs[i][j] = d
            if i == 3:
                print(":: All divergences will complete in {:} seconds".format(
                    (time.time() - t_start) / 3 * self.nb_models))

        return divs


def main():
    print("********************************************************")
    print("Welcome to the divergence computing tool for statistical")
    print("       texture content retrieval applications")
    print("       Developed and maintained by Hassan Rami")
    print("             hassan.rami@outlook.com")
    print("                   Version 1.0")
    print("           Last update: September 2019")
    print("********************************************************")

    """ Read command line arguments """
    parser = argparse.ArgumentParser(description="Compute divergences between statistical models")
    parser.add_argument('-f', metavar='models_file', help='Models file name')
    parser.add_argument('-m', metavar='model', help='Model type (ggd, weibull, ...)')
    parser.add_argument('-d', '--div', metavar='divergence_type', default='kld', help='Divergence type (kld or csd)')
    parser.add_argument('-gpu', type=str2bool, default=True, help='Use accelerated computation using GPU programming')
    parser.add_argument('-mci', type=str2bool, default=False, help='Use Monte-Carlo integration for non-analytic divergences')
    parser.add_argument('-s', '--save', type=bool, default=True, help='Save results in a text file')
    args = parser.parse_args()

    if args.m == 'weibull':
        model_set = WeibullModelSet(args.f, args.m, args.div, args.mci, args.gpu)
    elif args.m == 'ggd':
        model_set = GGDModelSet(args.f, args.m, args.div, args.mci, args.gpu)
    else:
        print('Model type {:} not supported!'.format(args.m))
        raise Exception('Model type not supported!')

    divs, arr = model_set.evaluate()

    # Save divergences in a csv file
    if args.save:
        divs_filename = args.f.replace("models", "divs")
        with open(divs_filename, mode="w+", newline='') as csvfile:
            fw = csv.writer(csvfile, delimiter=',')
            for i in range(model_set.nb_models):
                fw.writerow(divs[i])


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    main()

