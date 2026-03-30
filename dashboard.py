import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="QML Healthcare Dashboard", layout="wide")

st.title("Quantum Machine Learning for IoMT Healthcare Systems")
st.markdown("Interactive dashboard reproducing results and providing comparisons between Classical and Quantum ML models.")

results_file = 'experiment_results.json'

if not os.path.exists(results_file):
    st.warning(f"Results file '{results_file}' not found. Please run `python run_experiments.py` first.")
    st.stop()

with open(results_file, 'r') as f:
    results = json.load(f)

datasets = list(results.keys())
noise_types = ['Bit-flip', 'Phase-flip', 'Depolarizing', 'Amplitude Damping', 'Phase Damping']
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
algorithms = ['UU_dag', 'Var_UU_dag', 'UU_QNN']

# Hardcoded results from the research paper (Table I) for comparison
paper_results = {
    '5G-SA': {
        'SVM': {'Accuracy': 0.85, 'Precision': 0.55, 'Recall': 0.65, 'F1 Score': 0.55},
        'ANN': {'Accuracy': 0.87, 'Precision': 0.67, 'Recall': 0.56, 'F1 Score': 0.64},
        'UU_dag': {'Accuracy': 0.58, 'Precision': 0.85, 'Recall': 0.70, 'F1 Score': 0.78},
        'Var_UU_dag': {'Accuracy': 0.72, 'Precision': 0.40, 'Recall': 0.30, 'F1 Score': 0.63},
        'UU_QNN': {'Accuracy': 1.00, 'Precision': 1.00, 'Recall': 1.00, 'F1 Score': 1.00}
    },
    'L5G1.0': {
        'SVM': {'Accuracy': 0.65, 'Precision': 0.65, 'Recall': 0.55, 'F1 Score': 0.64},
        'ANN': {'Accuracy': 0.78, 'Precision': 0.78, 'Recall': 0.76, 'F1 Score': 0.78},
        'UU_dag': {'Accuracy': 0.52, 'Precision': 1.00, 'Recall': 0.52, 'F1 Score': 0.69},
        'Var_UU_dag': {'Accuracy': 0.54, 'Precision': 1.00, 'Recall': 0.55, 'F1 Score': 0.71},
        'UU_QNN': {'Accuracy': 0.92, 'Precision': 0.89, 'Recall': 0.92, 'F1 Score': 0.91}
    },
    'WE20': {
        'SVM': {'Accuracy': 0.99, 'Precision': 0.99, 'Recall': 0.99, 'F1 Score': 0.99},
        'ANN': {'Accuracy': 1.00, 'Precision': 0.97, 'Recall': 0.95, 'F1 Score': 0.96},
        'UU_dag': {'Accuracy': 0.98, 'Precision': 1.00, 'Recall': 0.55, 'F1 Score': 0.50},
        'Var_UU_dag': {'Accuracy': 0.99, 'Precision': 0.85, 'Recall': 0.85, 'F1 Score': 0.91},
        'UU_QNN': {'Accuracy': 1.00, 'Precision': 1.00, 'Recall': 1.00, 'F1 Score': 1.00}
    },
    'PS-IoT': {
        'SVM': {'Accuracy': 0.82, 'Precision': 0.82, 'Recall': 0.93, 'F1 Score': 0.89},
        'ANN': {'Accuracy': 0.83, 'Precision': 0.82, 'Recall': 0.96, 'F1 Score': 0.89},
        'UU_dag': {'Accuracy': 0.54, 'Precision': 0.54, 'Recall': 0.54, 'F1 Score': 0.65},
        'Var_UU_dag': {'Accuracy': 0.84, 'Precision': 0.80, 'Recall': 0.99, 'F1 Score': 0.89},
        'UU_QNN': {'Accuracy': 0.55, 'Precision': 0.80, 'Recall': 0.56, 'F1 Score': 0.67}
    }
}

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview & Table I", 
    "Paper vs Implementation",
    "Noise Robustness (Paper Figs)", 
    "Algorithm Comparison", 
    "Metric Heatmaps"
])

if page == "Overview & Table I":
    st.header("Table I: Comparison of Performance Metrics")
    st.markdown("Comparison between Classical methods (SVM, ANN) and Quantum methods (UU†, Variational UU†, UU†-QNN).")
    
    table_data = []
    for ds in datasets:
        for cl_algo in ['SVM', 'ANN']:
            metrics = results[ds]['classical'][cl_algo]
            table_data.append({'Dataset': ds, 'Algorithm': cl_algo, 
                               'Accuracy': metrics['Accuracy'], 'Precision': metrics['Precision'], 
                               'Recall': metrics['Recall'], 'F1 Score': metrics['F1']})
        
        for q_algo in algorithms:
            metrics = results[ds]['quantum_clean'][q_algo]
            table_data.append({'Dataset': ds, 'Algorithm': q_algo, 
                               'Accuracy': metrics['Accuracy'], 'Precision': metrics['Precision'], 
                               'Recall': metrics['Recall'], 'F1 Score': metrics['F1']})
            
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True)
    
    st.subheader("Performance Bar Charts")
    sel_metric = st.selectbox("Select Metric to Visualize", ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    fig = px.bar(df_table, x="Dataset", y=sel_metric, color="Algorithm", barmode="group",
                 title=f"{sel_metric} Comparison Across Datasets")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Paper vs Implementation":
    st.header("Paper vs Implementation Comparison")
    st.markdown("Comparing the outcomes achieved in our current local simulation against the original values reported in Table I of the research paper.")
    
    # Build comparison dataframe
    comp_data = []
    for ds in datasets:
        # Classical
        for algo in ['SVM', 'ANN']:
            impl_metrics = results[ds]['classical'][algo]
            if ds in paper_results and algo in paper_results[ds]:
                paper_m = paper_results[ds][algo]
                comp_data.append({'Dataset': ds, 'Algorithm': algo, 'Source': 'Achieved',
                                  'Accuracy': impl_metrics['Accuracy'], 'F1 Score': impl_metrics['F1']})
                comp_data.append({'Dataset': ds, 'Algorithm': algo, 'Source': 'Paper',
                                  'Accuracy': paper_m['Accuracy'], 'F1 Score': paper_m['F1 Score']})
        
        # Quantum
        for algo in algorithms:
            impl_metrics = results[ds]['quantum_clean'][algo]
            if ds in paper_results and algo in paper_results[ds]:
                paper_m = paper_results[ds][algo]
                comp_data.append({'Dataset': ds, 'Algorithm': algo, 'Source': 'Achieved',
                                  'Accuracy': impl_metrics['Accuracy'], 'F1 Score': impl_metrics['F1']})
                comp_data.append({'Dataset': ds, 'Algorithm': algo, 'Source': 'Paper',
                                  'Accuracy': paper_m['Accuracy'], 'F1 Score': paper_m['F1 Score']})

    df_comp = pd.DataFrame(comp_data)
    
    comp_metric = st.selectbox("Select Metric for Comparison", ['Accuracy', 'F1 Score'])
    
    # Plotly bar charts split by dataset
    fig = px.bar(df_comp, x="Algorithm", y=comp_metric, color="Source", barmode="group",
                 facet_col="Dataset", facet_col_wrap=2, 
                 title=f"{comp_metric}: Implementation vs Research Paper",
                 height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Comparison Data Table")
    # Pivot for easier reading
    df_pivot = df_comp.pivot_table(index=['Dataset', 'Algorithm'], columns='Source', values=comp_metric).reset_index()
    df_pivot['Difference'] = df_pivot['Achieved'] - df_pivot['Paper']
    st.dataframe(df_pivot.style.background_gradient(subset=['Difference'], cmap='RdYlGn'), use_container_width=True)

elif page == "Noise Robustness (Paper Figs)":
    st.header("Noise Robustness Analysis")
    st.markdown("Replicating graphs similar to Figures 3, 4, and 5 from the paper. These plots showcase how accuracy degrades under varying levels of quantum noise.")
    
    sel_algo = st.selectbox("Select Algorithm", algorithms)
    
    cols = st.columns(2)
    for i, ds in enumerate(datasets):
        fig = go.Figure()
        
        for nt in noise_types:
            acc_list = results[ds]['quantum_noisy'][nt][sel_algo]
            fig.add_trace(go.Scatter(x=noise_levels, y=acc_list, mode='lines+markers', name=nt))
            
        fig.update_layout(title=f"Accuracy vs Noise for {sel_algo} on {ds}",
                          xaxis_title="Noise Parameter (γ)",
                          yaxis_title="Accuracy",
                          legend_title="Noise Model",
                          yaxis=dict(range=[0, 1.05]))
        cols[i % 2].plotly_chart(fig, use_container_width=True)

elif page == "Algorithm Comparison":
    st.header("Algorithm Comparison under Specific Noise")
    c1, c2 = st.columns(2)
    sel_ds = c1.selectbox("Select Dataset", datasets)
    sel_noise = c2.selectbox("Select Noise Model", noise_types)
    
    fig = go.Figure()
    for algo in algorithms:
        acc_list = results[sel_ds]['quantum_noisy'][sel_noise][algo]
        fig.add_trace(go.Scatter(x=noise_levels, y=acc_list, mode='lines+markers', name=algo))
        
    fig.update_layout(title=f"Comparison on {sel_ds} under {sel_noise} noise",
                      xaxis_title="Noise Parameter (γ)",
                      yaxis_title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Metric Heatmaps":
    st.header("Heatmaps of Performance Metrics")
    metric = st.selectbox("Metric", ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    heatmap_data = []
    for ds in datasets:
        row = {'Dataset': ds}
        for algo in ['SVM', 'ANN']:
            row[algo] = results[ds]['classical'][algo][metric]
        for algo in algorithms:
            row[algo] = results[ds]['quantum_clean'][algo][metric]
        heatmap_data.append(row)
        
    df_heat = pd.DataFrame(heatmap_data).set_index('Dataset')
    
    fig = px.imshow(df_heat, text_auto=True, color_continuous_scale="Viridis",
                    title=f"{metric} Heatmap (Darker = Better) wait Viridis lighter is higher")
    st.plotly_chart(fig, use_container_width=True)
