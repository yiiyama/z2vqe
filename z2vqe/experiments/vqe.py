"""Perform VQE."""
import numpy as np
import jax
import jax.numpy as jnp
from z2vqe.qfim_vqe import make_cost_fn, vqe

MASS = 0.6
COUPLING = 1.2


def perform_vqe(generators, ham_terms, subspace, num_layers, instances_per_device, seed=0):
    rng = np.random.default_rng(seed=seed)

    initial_state = np.zeros(subspace.shape[0], dtype=np.complex128)
    initial_state[0] = 1.
    initial_state = subspace.conjugate().T @ initial_state

    hamiltonian = jnp.sum(ham_terms[0:2], axis=0)
    hamiltonian += MASS * np.sum(ham_terms[2:4], axis=0)
    hamiltonian += COUPLING * np.sum(ham_terms[4:], axis=0)
    exact_e0 = np.linalg.eigvalsh(hamiltonian)[0]

    cost_fn = make_cost_fn(generators, num_layers)
    num_dev = jax.device_count()
    num_params = generators.shape[0] * num_layers
    if num_dev > 1:
        shape = (num_dev, instances_per_device, num_params)
    else:
        shape = (instances_per_device, num_params)
    params_init = rng.uniform(0., np.inf, size=shape)
    energies, _ = vqe(cost_fn, initial_state, hamiltonian, params_init, 10000, tol=0.,
                      target=exact_e0 + 1.e-6)
    print('VQE:', energies[0, -1], '-', exact_e0, '=', energies[0, -1] - exact_e0)
    return energies, exact_e0


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
        hva_gens = -1.j * source['hva_gen_proj'][()]
        subspace_proj = source['subspace'][()]

    filename = f'generators-gsp_msp_hsp-{options.num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        hamiltonian_terms = source['hva_gen_proj'][()]

    filename = f'qfim-{options.config}-{options.num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        nl = source['ranks'].shape[0] + 5  # pylint: disable=no-member

    losses, minimum = perform_vqe(hva_gens, hamiltonian_terms, subspace_proj, nl, 64)

    filename = f'vqe-{options.config}-{options.num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('energies', data=losses)
        out.create_dataset('e0_exact', data=minimum)
