"""Compute the QFIM rank."""
import os
from pathlib import Path
import logging
import numpy as np
import h5py
import jax
from z2vqe.qfim_vqe import qfim_saturation


def compute_qfim_rank(generators, subspace, points_per_device):
    initial_state = np.zeros(subspace.shape[0], dtype=np.complex128)
    initial_state[[0, -1]] = np.sqrt(0.5)
    initial_state = subspace.conjugate().T @ initial_state

    ranks, num_layers = qfim_saturation(generators, initial_state, points_per_device)
    rsat, lcritical = get_rsat_lcritical(ranks, num_layers)
    print('QFIM:', rsat)
    print('Lc:', lcritical)
    return ranks, num_layers


def get_rsat_lcritical(ranks, num_layers):
    mean_ranks = np.mean(ranks, axis=1)
    rsat = mean_ranks[-1]
    lcritical = num_layers[np.nonzero(np.isclose(mean_ranks, rsat))[0][0]]
    return rsat, lcritical


def main(config, num_fermions, out_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    jax.config.update('jax_enable_x64', True)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators = source['gen_mats'][()]
        subspace = source['subspace'][()]

    qfim_ranks, num_layers = compute_qfim_rank(generators, subspace, 100)

    filename = f'qfim-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('ranks', data=qfim_ranks)
        out.create_dataset('num_layers', data=num_layers)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--out-dir')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    main(options.config, options.num_fermions, options.out_dir, options.log_level)
