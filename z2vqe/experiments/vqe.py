"""Compute the QFIM rank and perform VQE over evenly spaced numbers of layers."""
import os
from pathlib import Path
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from z2vqe.qfim_vqe import (make_ansatz_fn, make_qfim_fn, make_cost_fn, random_params, run_vqe,
                            qfim_rank)
from z2vqe.experiments.qfim import get_rsat_lcritical

LOG = logging.getLogger(__name__)


def compute_qfim_vqe(generators, nls, hamiltonian, subspace, num_instances, maxiter, acceleration):
    points_per_device = num_instances // jax.device_count()

    initial_state = np.zeros(subspace.shape[0], dtype=np.complex128)
    initial_state[[0, -1]] = np.sqrt(0.5)
    initial_state = subspace.conjugate().T @ initial_state

    qfim_ranks = np.zeros((len(nls), num_instances))
    gradients = np.zeros_like(qfim_ranks)
    losses = np.zeros_like(qfim_ranks)

    for inl, nl in enumerate(nls):
        LOG.info('Instance: %d layers (%d/%d)', nl, inl + 1, len(nls))
        num_params = generators.shape[0] * nl
        init_params = random_params(num_params, points_per_device, seed=123456)
        ansatz_fn = make_ansatz_fn(generators, nl)
        qfim_fn = make_qfim_fn(ansatz_fn=ansatz_fn, vmap=True, pmap=jax.device_count() > 1)
        cost_fn, grad_fn = make_cost_fn(ansatz_fn=ansatz_fn, with_grad=True, vmap=True,
                                        pmap=jax.device_count() > 1)

        @jax.jit
        def get_qfim_rank(init_params, initial_state):
            qfim = qfim_fn(init_params, initial_state)
            qfim = qfim.reshape((num_instances, num_params, num_params))
            svals = jnp.linalg.svd(qfim, compute_uv=False, hermitian=True)
            return qfim_rank(svals, atol=1.e-10, rtol=1.e-6, npmod=jnp)

        @jax.jit
        def get_dldt(init_params, initial_state, hamiltonian):
            grad = grad_fn(init_params, initial_state, hamiltonian)
            grad = grad.reshape((num_instances, num_params))
            return jnp.sum(jnp.square(grad), axis=1)

        qfim_ranks[inl] = get_qfim_rank(init_params, initial_state)
        gradients[inl] = get_dldt(init_params, initial_state, hamiltonian)
        if acceleration:
            energies = run_vqe(cost_fn, initial_state, hamiltonian, init_params, maxiter)[0]
        else:
            energies = run_vqe(cost_fn, initial_state, hamiltonian, init_params, maxiter,
                               stepsize=1.e-4, acceleration=False)[0]
        losses[inl] = energies[:, -1]

    return qfim_ranks, gradients, losses


def main(config, num_fermions, num_instances, maxiter, out_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    jax.config.update('jax_enable_x64', True)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators = source['gen_mats'][()]
        subspace = source['subspace'][()]
        hamiltonian = source['hamiltonian'][()]

    filename = f'qfim-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        ranks = source['ranks'][()]
        num_layers = source['num_layers'][()]

    lcr = get_rsat_lcritical(ranks, num_layers)[1]
    nlstep = 2 * max(1, np.round(np.log(lcr)).astype(int))
    nls_low = np.arange(lcr, nlstep, -nlstep)[::-1]
    if len(nls_low) < 2:
        nls_low = np.unique([2, lcr])
    nls = np.concatenate([nls_low, np.arange(lcr + nlstep, lcr + 5 * nlstep, nlstep)])
    print(lcr, nlstep, nls)
    filename = f'vqe-{config}-{num_fermions}.h5'

    acceleration = num_fermions != 2
    if not acceleration:
        maxiter *= 10

    ranks, gradients, losses = compute_qfim_vqe(generators, nls, hamiltonian, subspace,
                                                num_instances, maxiter=maxiter,
                                                acceleration=acceleration)

    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('num_layers', data=nls)
        out.create_dataset('ranks', data=ranks)
        out.create_dataset('gradients', data=gradients)
        out.create_dataset('losses', data=losses)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--instances', type=int, default=1024)
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--out-dir')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'

    main(options.config, options.num_fermions, options.instances, options.maxiter, options.out_dir,
         options.log_level)
