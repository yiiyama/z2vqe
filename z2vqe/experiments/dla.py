"""Compute the DLA of the HVA generators."""
import os
from pathlib import Path
import logging
import h5py
import jax
import jax.numpy as jnp
from fastdla.lie_closure import lie_closure


def compute_dla(generators):
    dla = lie_closure(jnp.array(generators), max_dim=100000, skew_hermitian=True)
    print('DLA:', dla.shape[0])
    return dla


def main(config, num_fermions, out_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    jax.config.update('jax_enable_x64', True)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators_herm = source['hva_gen_proj'][()]

    dla = compute_dla(-1.j * generators_herm)

    filename = f'dla-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('dla', data=dla)


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
