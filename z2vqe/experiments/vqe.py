"""Perform VQE."""
import os
from pathlib import Path
import logging
import numpy as np
import h5py
import jax
from z2vqe.qfim_vqe import vqe
from z2vqe.experiments.qfim import get_rsat_lcritical


def perform_vqe(generators, num_layers, hamiltonian, subspace, instances_per_device):
    initial_state = np.zeros(subspace.shape[0], dtype=np.complex128)
    initial_state[[0, -1]] = np.sqrt(0.5)
    initial_state = subspace.conjugate().T @ initial_state

    exact_e0 = np.linalg.eigvalsh(hamiltonian)[0]
    energies, _ = vqe(generators, num_layers, initial_state, hamiltonian,
                      instances_per_device=instances_per_device, tol=0., target=exact_e0 + 1.e-6)

    print('VQE:', energies[0, -1], '-', exact_e0, '=', energies[0, -1] - exact_e0)
    return energies, exact_e0


def main(config, num_fermions, out_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    jax.config.update('jax_enable_x64', True)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators = source['gen_mats'][()]
        subspace = source['subspace'][()]
        hamiltonian = source['hamiltonian'][()]

    filename = f'qfim-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        ranks = source['ranks'][()]
        num_layers = source['num_layers'][()]

    nl = get_rsat_lcritical(ranks, num_layers)[1] + 5
    losses, minimum = perform_vqe(generators, hamiltonian, subspace, nl, 64)

    filename = f'vqe-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('energies', data=losses)
        out.create_dataset('e0_exact', data=minimum)


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
