"""Perform simultaneous block diagonalization of the HVA generators."""
from pathlib import Path
import logging
import numpy as np
import h5py
from fastdla.linalg.block_diagonalization import sbd_fast


def block_diagonalize(generators_herm, npmod=np):
    tr, blocks = sbd_fast(generators_herm, hermitian=True, return_blocks=True, npmod=npmod)
    print('blocks:', [block.shape[1] for block in blocks])
    return tr, blocks


def main(config, num_fermions, out_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators = source['hva_gen_proj'][()]

    transform, blocks = block_diagonalize(generators)

    filename = f'sbd-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('transform', data=transform)
        for iblock, block in enumerate(blocks):
            out.create_dataset(f'block{iblock}', data=block)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--out-dir')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    main(options.config, options.num_fermions, options.out_dir, options.log_level)
