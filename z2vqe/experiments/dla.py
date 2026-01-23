"""Compute the DLA of the HVA generators."""
import jax
import jax.numpy as jnp
from fastdla.lie_closure import lie_closure


def compute_dla(generators):
    closure = lie_closure(jnp.array(generators), max_dim=100000, skew_hermitian=True)
    print('DLA:', closure.shape[0])
    return closure


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    from pathlib import Path
    import logging
    import h5py

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--out-dir')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))
    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    jax.config.update('jax_enable_x64', True)

    out_dir = Path(options.out_dir or '.')
    filename = f'generators-{options.config}-{options.num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators_herm = source['hva_gen_proj'][()]

    dla = compute_dla(-1.j * generators_herm)

    filename = f'dla-{options.config}-{options.num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('dla', data=dla)
