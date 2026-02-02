"""Compute the DLA of the HVA generators."""
import os
from pathlib import Path
import time
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from fastdla.lie_closure import lie_closure


def compute_dla(generators_herm, distributed=False, shard_template=None):
    shard_basis = jax.device_count() > 1
    dla = lie_closure(jnp.array(-1.j * generators_herm), skew_hermitian=True,
                      shard_basis=shard_basis)
    
    if shard_basis:
        basis, indices = dla
        if distributed:
            proc_id = jax.process_index()
            local_indices = indices[1][indices[0] == proc_id]
            local_shard = basis.addressable_shards[0].data[0, local_indices]
            print(shard_template % proc_id)
            with h5py.File(shard_template % proc_id, 'w', libver='latest') as out:
                out.create_dataset('dla_shard', data=local_shard)

            if proc_id == 0:
                # We assume one device per process
                dla = [None,] * jax.device_count()
                pids = list(range(jax.device_count()))
                while pids:
                    pid = pids.pop()
                    try:
                        with h5py.File(shard_template % pid, 'r', libver='latest') as source:
                            dla[pid] = source['dla_shard'][()]
                        os.unlink(shard_template % pid)
                    except (FileNotFoundError, BlockingIOError):
                        pids.append(pid)
                        time.sleep(2)
                dla = np.concatenate(dla, axis=0)
            else:
                return None
        else:
            dla = np.array(basis)[indices]

    print('DLA:', dla.shape[0])
    return dla


def compute_block_dla(blocks, eigvecs, distributed=False, shard_template=None):
    # Which column has the state |00..0> ?
    col = np.argmax(np.abs(eigvecs[0]))
    iblock = np.searchsorted(np.cumsum([block.shape[1] for block in blocks]), col, side='right')
    print('Block:', blocks[iblock].shape[1])
    return compute_dla(blocks[iblock], distributed=distributed, shard_template=shard_template)


def main(config, num_fermions, out_dir, log_level, distributed=False):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    jax.config.update('jax_enable_x64', True)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators_herm = source['hva_gen_proj'][()]
        subspace = source['subspace'][()]
        u1_eigenidx = source['u1_eigenidx'][()]

    shard_template = str(out_dir / f'dlashard-{config}-{num_fermions}-%d.h5')
    dla = compute_dla(generators_herm, distributed=distributed, shard_template=shard_template)

    block_dla = None
    filename = f'sbd-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        if 'block1' in source:
            tr = source['transform'][()]
            blocks = [source[f'block{i}'][()] for i in range(len(source.keys()) - 1)]
            eigvecs = np.empty((2 ** (2 * num_fermions), tr.shape[1]), dtype=np.complex128)
            eigvecs[u1_eigenidx] = subspace @ tr
            block_dla = compute_block_dla(blocks, eigvecs, distributed=distributed,
                                          shard_template=shard_template)

    if jax.process_index() == 0:
        filename = f'dla-{config}-{num_fermions}.h5'
        with h5py.File(out_dir / filename, 'w', libver='latest') as out:
            out.create_dataset('dla', data=dla)
            if block_dla is not None:
                out.create_dataset('block_dla', data=block_dla)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--out-dir')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    if (gpu := options.gpu):
        if gpu == 'mpi':
            jax.distributed.initialize(cluster_detection_method="mpi4py")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    print(jax.devices())
    main(options.config, options.num_fermions, options.out_dir, options.log_level,
         distributed=gpu == 'mpi')
