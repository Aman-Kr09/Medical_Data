from qiskit_aer.noise import (NoiseModel, pauli_error, depolarizing_error,
                              amplitude_damping_error, phase_damping_error)

def get_noise_model(noise_type, p):
    if p == 0:
        return None
        
    noise_model = NoiseModel()
    
    if noise_type == 'Bit-flip':
        error_1q = pauli_error([('X', p), ('I', 1 - p)])
        error_2q = error_1q.tensor(error_1q)
    elif noise_type == 'Phase-flip':
        error_1q = pauli_error([('Z', p), ('I', 1 - p)])
        error_2q = error_1q.tensor(error_1q)
    elif noise_type == 'Depolarizing':
        error_1q = depolarizing_error(p, 1)
        error_2q = depolarizing_error(p, 2)
    elif noise_type == 'Amplitude Damping':
        error_1q = amplitude_damping_error(p)
        error_2q = error_1q.tensor(error_1q)
    elif noise_type == 'Phase Damping':
        error_1q = phase_damping_error(p)
        error_2q = error_1q.tensor(error_1q)
    else:
        return None
        
    noise_model.add_all_qubit_quantum_error(error_1q, ['u', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    
    return noise_model
