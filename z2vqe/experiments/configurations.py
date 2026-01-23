"""Experiment configurations."""

CONFIGS = {
    'hamiltonian': (([0, 1], [2, 3], [4, 5]), dict(u1_charge=0)),
    'g_m_h': (([0, 1], [2, 3], [4, 5]), dict(u1_charge=0, c_phase=0, p_sign=1)),
    'g_m_hsp': (([0, 1], [2, 3], [4], [5]), dict(u1_charge=0, t2_momentum=0, cp_sign=1)),
    'gsp_m_h': (([0], [1], [2, 3], [4, 5]), dict(u1_charge=0, t2_momentum=0, p_sign=1)),
    'g_h': (([0, 1], [4, 5]), dict(u1_charge=0, c_phase=0, p_sign=1)),
    'm_h': (([2, 3], [4, 5]), dict(u1_charge=0, c_phase=0, p_sign=1)),
}
