"""Experiment functions."""
from .generators import main as generators
from .sbd import main as sbd
from .qfim import main as qfim
from .vqe import main as vqe

experiments = {
    'generators': generators,
    'sbd': sbd,
    'qfim': qfim,
    'vqe': vqe
}
