"""Compute the DLA of the HVA generators."""
import os
from pathlib import Path
import logging
import h5py
import jax
import jax.numpy as jnp
from fastdla.lie_closure import lie_closure


def compute_dla(generators_herm):  # , distributed=False, shard_template=None):
    shard_basis = jax.device_count() > 1
    result = lie_closure(jnp.array(1.j * generators_herm), real=True, shard_basis=shard_basis)

    if shard_basis:
        dla_is, indices_is, dla_ra, indices_ra = result
        print('DLA:', indices_is[0].shape[0] + indices_ra[0].shape[0])
        return indices_is[0].shape[0] + indices_ra[0].shape[0]
    #     basis, indices = dla
    #     if distributed:
    #         proc_id = jax.process_index()
    #         local_indices = indices[1][indices[0] == proc_id]
    #         local_shard = basis.addressable_shards[0].data[0, local_indices]
    #         print(shard_template % proc_id)
    #         with h5py.File(shard_template % proc_id, 'w', libver='latest') as out:
    #             out.create_dataset('dla_shard', data=local_shard)

    #         if proc_id == 0:
    #             # We assume one device per process
    #             dla = [None,] * jax.device_count()
    #             pids = list(range(jax.device_count()))
    #             while pids:
    #                 pid = pids.pop()
    #                 try:
    #                     with h5py.File(shard_template % pid, 'r', libver='latest') as source:
    #                         dla[pid] = source['dla_shard'][()]
    #                     os.unlink(shard_template % pid)
    #                 except (FileNotFoundError, BlockingIOError):
    #                     pids.append(pid)
    #                     time.sleep(2)
    #             dla = np.concatenate(dla, axis=0)
    #         else:
    #             return None
    #     else:
    #         dla = np.array(basis)[indices]

    dla_is, dla_ra = result
    print('DLA:', dla_is.shape[0] + dla_ra.shape[0])
    return dla_is.shape[0] + dla_ra.shape[0]


def main(config, num_fermions, out_dir, log_level):  # , distributed=False):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    jax.config.update('jax_enable_x64', True)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators = source['gen_mats'][()]

    # shard_template = str(out_dir / f'dlashard-{config}-{num_fermions}-%d.h5')
    dla_dim = compute_dla(generators)  # , distributed=distributed, shard_template=shard_template)

    if jax.process_index() == 0:
        filename = f'dla-{config}-{num_fermions}.h5'
        with h5py.File(out_dir / filename, 'w', libver='latest') as out:
            out.create_dataset('dla_dim', data=dla_dim)


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

    main(options.config, options.num_fermions, options.out_dir, options.log_level)
