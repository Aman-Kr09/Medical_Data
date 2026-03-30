import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def load_data(dataset_name, n_samples=300):
    # n_samples is reduced for fast simulation. Paper used larger sets, 
    # but due to QASM simulator constraints locally we'll default to a smaller batch,
    # and we can scale it up if needed.
    if dataset_name == '5G-SA':
        n_features = 14
        m_qubits = 4 # ceil(log2(14))
    elif dataset_name == 'L5G1.0':
        n_features = 13
        m_qubits = 4
    elif dataset_name == 'WE20':
        n_features = 33
        m_qubits = 6 # ceil(log2(33)) = 6
    elif dataset_name == 'PS-IoT':
        n_features = 5
        m_qubits = 3 # ceil(log2(5))
    else:
        raise ValueError("Unknown dataset")
        
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_features-1, n_redundant=0, 
                               n_classes=2, class_sep=3.0, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means to get centroids
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)
    centroids = kmeans.cluster_centers_
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, centroids, m_qubits
