# pylint: disable=wrong-import-position, invalid-name
"""Run the VQE for Z2 LGT."""
import sys
import os
from argparse import ArgumentParser
import numpy as np
import jax
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from z2_lgt import initial_state, z2_ansatz_layer, create_hamiltonian, calculate_num_params
from z2_vqe import make_cost_fn, vqe_jaxopt


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    ####################################################################
    parser = ArgumentParser()
    parser.add_argument('sites', type=int)
    parser.add_argument('layers', type=int)
    parser.add_argument('j_hopping', type=float)
    parser.add_argument('f_gauge', type=float)
    parser.add_argument('mass', type=float)
    parser.add_argument('-i', '--maxiter', type=int, default=10000)
    parser.add_argument('-s', '--stepsize', type=float, default=0.)
    parser.add_argument('-c', '--instances', type=int, default=1)
    parser.add_argument('-r', '--seed', type=int)
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default=['0'])
    parser.add_argument('-o', '--out')
    options = parser.parse_args()

    ####################################################################

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(options.gpus)
    num_devices = len(jax.devices())
    if options.instances % num_devices != 0:
        raise ValueError(f'Number of instances {options.instances} must be a multiple of the number'
                         f' of GPUs {num_devices}')

    trans_inv = True
    overall_coeff = 10
    boundary_cond = 'closed'
    reference_values = {}

    init_state, _ = initial_state(options.sites, boundary_cond)
    ansatz_layer = z2_ansatz_layer(options.sites, boundary_cond)
    hamiltonian = create_hamiltonian(options.sites, options.j_hopping, options.f_gauge,
                                     options.mass, overall_coeff, boundary_cond,
                                     overall_coeff_cond=False)

    cost_fn = make_cost_fn(init_state, ansatz_layer, options.layers, hamiltonian)

    num_parameters = calculate_num_params(options.sites, options.layers, trans_inv)
    instances_per_device = max(1, options.instances // num_devices)
    rng = np.random.default_rng(options.seed)
    x0 = 2 * np.pi * rng.random((num_devices, instances_per_device, num_parameters))

    energies, parameters = vqe_jaxopt(cost_fn, x0, options.maxiter, stepsize=options.stepsize)

    if not options.out:
        options.out = (f'vqe_{options.sites}sites_{options.layers}layers_'
                       f'{options.maxiter}iter_jaxopt.h5')

    with h5py.File(options.out, 'a') as out:
        group = out.create_group(f'vqe_{len(out.keys())}')
        group.create_dataset('num_sites', data=options.sites)
        group.create_dataset('num_layers', data=options.layers)
        group.create_dataset('j_hopping', data=options.j_hopping)
        group.create_dataset('f_gauge', data=options.f_gauge)
        group.create_dataset('mass', data=options.mass)
        group.create_dataset('maxiter', data=options.maxiter)
        group.create_dataset('stepsize', data=options.stepsize)
        group.create_dataset('x0', data=x0.reshape(-1, num_parameters))
        group.create_dataset('energies', data=energies)
        group.create_dataset('parameters', data=parameters)
