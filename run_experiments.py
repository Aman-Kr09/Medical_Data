import json
import time
import pandas as pd
from data_loader import load_data
from train_evaluate import *
from qiskit_aer import AerSimulator

def run_experiments():
    datasets = ['5G-SA', 'L5G1.0', 'WE20', 'PS-IoT']
    noise_types = ['Bit-flip', 'Phase-flip', 'Depolarizing', 'Amplitude Damping', 'Phase Damping']
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    simulator = AerSimulator()
    results = {}
    
    try:
        for ds in datasets:
            print(f"Running experiments for {ds}...", flush=True)
            results[ds] = {'classical': {}, 'quantum_clean': {}, 'quantum_noisy': {}}
            
            # Using a small batch of 30 samples (20 train, 10 test) for demonstration purposes.
            # This keeps quantum simulation time feasible for local machines.
            X_train, X_test, y_train, y_test, centroids, m_qubits = load_data(ds, n_samples=30)
            
            class_res = evaluate_classical(X_train, y_train, X_test, y_test)
            results[ds]['classical'] = class_res
            
            print(f"  Training UU_QNN for {ds}...", flush=True)
            qnn_params = train_uu_qnn(X_train, y_train, centroids, m_qubits, simulator)
            
            print(f"  Evaluating Clean Quantum models for {ds}...", flush=True)
            y_pred_uu = predict_uu_dag(X_test, centroids, m_qubits, simulator)
            y_pred_var = predict_uu_dag(X_test, centroids, m_qubits, simulator, var=True)
            y_pred_qnn = predict_uu_qnn(X_test, centroids, m_qubits, qnn_params, simulator)
            
            results[ds]['quantum_clean'] = {
                'UU_dag': evaluate_metrics(y_test, y_pred_uu),
                'Var_UU_dag': evaluate_metrics(y_test, y_pred_var),
                'UU_QNN': evaluate_metrics(y_test, y_pred_qnn)
            }
            
            print(f"  Evaluating Noisy Quantum models for {ds}...", flush=True)
            for nt in noise_types:
                results[ds]['quantum_noisy'][nt] = {'UU_dag': [], 'Var_UU_dag': [], 'UU_QNN': []}
                for p in noise_levels:
                    if p == 0:
                        results[ds]['quantum_noisy'][nt]['UU_dag'].append(results[ds]['quantum_clean']['UU_dag']['Accuracy'])
                        results[ds]['quantum_noisy'][nt]['Var_UU_dag'].append(results[ds]['quantum_clean']['Var_UU_dag']['Accuracy'])
                        results[ds]['quantum_noisy'][nt]['UU_QNN'].append(results[ds]['quantum_clean']['UU_QNN']['Accuracy'])
                        continue
                    
                    noise_model = get_noise_model(nt, p)
                    y_p_uu = predict_uu_dag(X_test, centroids, m_qubits, simulator, noise_model=noise_model)
                    y_p_var = predict_uu_dag(X_test, centroids, m_qubits, simulator, noise_model=noise_model, var=True)
                    y_p_qnn = predict_uu_qnn(X_test, centroids, m_qubits, qnn_params, simulator, noise_model=noise_model)
                    
                    results[ds]['quantum_noisy'][nt]['UU_dag'].append(evaluate_metrics(y_test, y_p_uu)['Accuracy'])
                    results[ds]['quantum_noisy'][nt]['Var_UU_dag'].append(evaluate_metrics(y_test, y_p_var)['Accuracy'])
                    results[ds]['quantum_noisy'][nt]['UU_QNN'].append(evaluate_metrics(y_test, y_p_qnn)['Accuracy'])
                    
        with open('experiment_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        print("Experiments completed. Results saved to experiment_results.json", flush=True)
    except Exception as e:
        import traceback
        with open('error_log.txt', 'w') as f:
            f.write(traceback.format_exc())
        print(f"FAILED: {e}")


if __name__ == "__main__":
    run_experiments()
