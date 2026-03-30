import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from quantum_models import *
from noise_simulation import get_noise_model

def evaluate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }

def evaluate_classical(X_train, y_train, X_test, y_test):
    svm = SVC(kernel='linear') 
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    
    ann = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
    ann.fit(X_train, y_train)
    y_pred_ann = ann.predict(X_test)
    
    results = {
        'SVM': evaluate_metrics(y_test, y_pred_svm),
        'ANN': evaluate_metrics(y_test, y_pred_ann)
    }
    return results

def predict_uu_dag(X, centroids, m_qubits, simulator, noise_model=None, var=False):
    y_pred = []
    for x in X:
        if not var:
            qc0 = get_uu_dag_circuit(centroids[0], x, m_qubits)
            qc1 = get_uu_dag_circuit(centroids[1], x, m_qubits)
        else:
            qc0 = get_var_uu_dag_circuit(centroids[0], x, m_qubits)
            qc1 = get_var_uu_dag_circuit(centroids[1], x, m_qubits)
            
        p_c0 = run_circuit_prob_0(qc0, simulator, noise_model=noise_model)
        p_c1 = run_circuit_prob_0(qc1, simulator, noise_model=noise_model)
        
        y_pred.append(0 if p_c0 > p_c1 else 1)
    return y_pred

def train_uu_qnn(X_train, y_train, centroids, m_qubits, simulator):
    init_params = np.ones(2 * m_qubits) * 0.1 
    
    def qnn_loss(params):
        y_pred = predict_uu_qnn(X_train, centroids, m_qubits, params, simulator, noise_model=None)
        acc = accuracy_score(y_train, y_pred)
        return (1.0 - acc) ** 2

    cobyla = COBYLA(maxiter=20)
    result = cobyla.minimize(fun=qnn_loss, x0=init_params)
    return result.x

def predict_uu_qnn(X, centroids, m_qubits, params, simulator, noise_model=None):
    y_pred = []
    for x in X:
        qc0 = get_uu_qnn_circuit(centroids[0], x, m_qubits, params)
        qc1 = get_uu_qnn_circuit(centroids[1], x, m_qubits, params)
        p_c0 = run_circuit_prob_0(qc0, simulator, noise_model=noise_model)
        p_c1 = run_circuit_prob_0(qc1, simulator, noise_model=noise_model)
        y_pred.append(0 if p_c0 > p_c1 else 1)
    return y_pred
