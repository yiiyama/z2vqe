"""Run the QFIM calculations."""
import sys
import os
import time
from argparse import ArgumentParser
import numpy as np
import h5py
import jax
jax.config.update('jax_enable_x64', True)
sys.path.append('/home/iiyama/src/z2vqe/src')
from z2_lgt import calculate_num_params, z2_ansatz_layer, initial_state
from z2_vqe import make_qfim_fn

if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    ####################################################################
    parser = ArgumentParser()
    parser.add_argument('-s', '--sites', type=int, nargs='+')
    parser.add_argument('-l', '--layers', type=int, nargs='+')
    parser.add_argument('-p', '--points', type=int, default=1)
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default=['0'])
    parser.add_argument('-o', '--out')
    options = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, options.gpus))
    num_devices = len(jax.devices())
    if options.points % num_devices != 0:
        raise ValueError(f'Number of points {options.points} must be a multiple of the number'
                         f' of GPUs {num_devices}')

    trans_inv = True
    boundary_cond = 'closed'

    points_per_device = max(1, options.points // num_devices)

    if not options.out:
        options.out = 'qfim.h5'

    out = h5py.File(options.out, 'w')

    start_time = time.time()

    for num_sites in options.sites:
        init_state, _ = initial_state(num_sites, boundary_cond)
        ansatz_layer = z2_ansatz_layer(num_sites, boundary_cond)
        mean_ranks = []
        max_ranks = []
        for num_layers in options.layers:
            print(num_sites, 'sites', num_layers, 'layers')
            qfim_fn = jax.pmap(jax.vmap(make_qfim_fn(init_state, ansatz_layer, num_layers)))
            num_parameters = calculate_num_params(num_sites, num_layers, trans_inv)
            params = 2 * np.pi * np.random.random((num_devices, points_per_device, num_parameters))
            matrices = qfim_fn(params).reshape(-1, num_parameters, num_parameters)
            ranks = np.linalg.matrix_rank(matrices, tol=1.e-12, hermitian=True)

            group = out.create_group(f'qfim_{num_sites}sites_{num_layers}layers')
            group.create_dataset('params', data=params.reshape(-1, num_parameters))
            group.create_dataset('qfim', data=matrices)
            group.create_dataset('rank', data=ranks)

            mean_ranks.append(np.mean(ranks))
            max_ranks.append(np.amax(ranks))
            print('  rank mean', mean_ranks[-1])
            print('  rank max', max_ranks[-1])
            print('Elapsed time:', time.time() - start_time, 's')

            if (len(mean_ranks) > 4 and np.allclose(mean_ranks[-1], mean_ranks[-3:-1])
                    and np.allclose(max_ranks[-1], max_ranks[-3:-1])):
                break

    out.close()
