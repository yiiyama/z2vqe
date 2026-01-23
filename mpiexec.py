from argparse import ArgumentParser
from itertools import product
from mpi4py import MPI
from z2vqe.experiments import experiments

comm = MPI.COMM_WORLD

parser = ArgumentParser()
parser.add_argument('experiment')
parser.add_argument('--config', nargs='+')
parser.add_argument('--num-fermions', type=int, nargs='+')
parser.add_argument('--out-dir')
parser.add_argument('--log-level', default='warning')
options = parser.parse_args()

exp_conf = list(product(options.config, options.num_fermions))
config, num_fermions = exp_conf[comm]

exp_fn = experiments[options.experiment]
exp_fn(config, num_fermions, options.out_dir, options.log_level)
