"""Perform simultaneous block diagonalization of the HVA generators."""
import numpy as np
from fastdla.linalg.block_diagonalization import sbd_fast


def block_diagonalize(generators_herm, npmod=np):
    tr, blocks = sbd_fast(generators_herm, hermitian=True, return_blocks=True, npmod=npmod)
    print('blocks:', [block.shape[1] for block in blocks])
    return tr, blocks


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    import logging
    import h5py

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--out-dir')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    out_dir = Path(options.out_dir or '.')
    filename = f'generators-{options.config}-{options.num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators = source['hva_gen_proj'][()]

    transform, diag_blocks = block_diagonalize(generators)

    filename = f'sbd-{options.config}-{options.num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('transform', data=transform)
        for iblock, block in enumerate(diag_blocks):
            out.create_dataset(f'block{iblock}', data=block)
