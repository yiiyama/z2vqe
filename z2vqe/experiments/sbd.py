"""Perform simultaneous block diagonalization of the HVA generators."""
from pathlib import Path
import logging
import h5py
# import jax
# import jax.numpy as jnp
from fastdla.algorithms.block_diagonalization import sbd_fast


def block_diagonalize(generators_herm, orth_cutoff=1.e-08):
    tr, blocks = sbd_fast(generators_herm, hermitian=True, return_blocks=True,
                          orth_cutoff=orth_cutoff)
    print('blocks:', [block.shape[1] for block in blocks])
    return tr, blocks


def main(config, num_fermions, out_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    # jax.config.update('jax_enable_x64', True)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators = source['hva_gen_proj'][()]

    transform, blocks = block_diagonalize(generators, orth_cutoff=1.e-4)

    filename = f'sbd-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('transform', data=transform)
        for iblock, block in enumerate(blocks):
            out.create_dataset(f'block{iblock}', data=block)


if __name__ == '__main__':
    # import os
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--out-dir')
    parser.add_argument('--gpu')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    # if options.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    main(options.config, options.num_fermions, options.out_dir, options.log_level)
