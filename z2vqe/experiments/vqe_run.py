"""Perform VQE."""
import os
from pathlib import Path
import logging
import numpy as np
import h5py
import jax
from z2vqe.qfim_vqe import make_cost_fn, random_params, run_vqe
from z2vqe.experiments.qfim import get_rsat_lcritical

LOG = logging.getLogger(__name__)


def perform_vqe(generators, nls, hamiltonian, subspace, ntherm, maxiter, stepsize, dldtmax,
                instances_per_device):
    initial_state = np.zeros(subspace.shape[0], dtype=np.complex128)
    initial_state[[0, -1]] = np.sqrt(0.5)
    initial_state = subspace.conjugate().T @ initial_state

    exact_e0 = np.linalg.eigvalsh(hamiltonian)[0]

    num_instances = instances_per_device * jax.device_count()
    all_energies = np.zeros((len(nls), num_instances, ntherm + maxiter + 1))
    all_params = []
    all_stepsizes = np.zeros((len(nls), num_instances, ntherm))

    for inl, nl in enumerate(nls):
        LOG.info('VQE: %d layers (%d/%d)', nl, inl + 1, len(nls))
        cost_fn = make_cost_fn(generators, nl, vmap=True, pmap=jax.device_count() > 1)
        init_params = random_params(generators.shape[0] * nl, instances_per_device)
        if ntherm > 0:
            # pylint: disable=unbalanced-tuple-unpacking
            energies, params, stepsizes = run_vqe(cost_fn, initial_state, hamiltonian, init_params,
                                                  ntherm, dldtmax=dldtmax)
            all_energies[inl, :, :ntherm + 1] = energies
            all_params.append(np.pad(params[:, :ntherm + 1], [(0, 0), (0, maxiter), (0, 0)]))
            all_stepsizes[inl] = stepsizes
        else:
            params = init_params[:, None]
            all_params.append(
                np.empty((np.prod(init_params.shape[:2]), maxiter + 1, init_params.shape[2]))
            )
        if maxiter > 0:
            resume_params = params[:, -1].reshape(init_params.shape)
            energies, params = run_vqe(cost_fn, initial_state, hamiltonian, resume_params, maxiter,
                                       stepsize=stepsize, acceleration=False, dldtmax=dldtmax)[:2]
            all_energies[inl, :, ntherm:] = energies
            all_params[-1][:, ntherm:] = params

    return all_energies, exact_e0, all_params, all_stepsizes


def main(config, num_fermions, ntherm, maxiter, stepsize, dldtmax, nls, out_dir, ipd, save_params,
         log_level):
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
    if nls is None:
        # min_nl = max(lcr // 2, 4)
        # max_nl = lcr * 2
        # nls = np.round(np.linspace(min_nl, max_nl, 16)).astype(int)
        # matches = np.diff(nls) == 0
        # nls = nls[~np.concatenate([matches, [False]])]
        nlstep = 2 * max(1, np.round(np.log(lcr)).astype(int))
        nls = np.arange(4, lcr + 5 * nlstep, nlstep)
        filename = f'vqe-{config}-{num_fermions}'
    else:
        nls = np.array(nls)
        filename = f'vqe-{config}-{num_fermions}-nl{"_".join(str(n) for n in nls)}'
    filename += f'-th{ntherm}-it{maxiter}.h5'

    energies, exact_e0, params, stepsizes = perform_vqe(generators, nls, hamiltonian, subspace,
                                                        ntherm, maxiter, stepsize, dldtmax, ipd)

    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('num_layers', data=nls)
        out.create_dataset('energies', data=energies)
        out.create_dataset('exact_e0', data=exact_e0)
        out.create_dataset('therm_stepsizes', data=stepsizes)
        out.create_dataset('run_stepsize', data=stepsize)
        if save_params:
            group = out.create_group('params')
            for nl, vals in zip(nls, params):
                group.create_dataset(f'l{nl}', data=vals)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--maxiter', type=int)
    parser.add_argument('--ntherm', type=int, default=20)
    parser.add_argument('--stepsize', type=float, default=5.e-5)
    parser.add_argument('--dldtmax', type=float, default=-1.)
    parser.add_argument('--nl', type=int, nargs='+')
    parser.add_argument('--gpu')
    parser.add_argument('--per-device', type=int, default=16)
    parser.add_argument('--save-params', action='store_true')
    parser.add_argument('--out-dir')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'

    main(options.config, options.num_fermions, options.ntherm, options.maxiter, options.stepsize,
         options.dldtmax, options.nl, options.out_dir, options.per_device, options.save_params,
         options.log_level)
