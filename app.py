<<<<<<< HEAD
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Final Year Project", layout="wide")

st.title("Final Year Project 🚀")

st.write("App started successfully ✅")

dataset = st.selectbox("Select Dataset", [
    "benchmark_adult.csv",
    "finance_credit.csv",
    "healthcare_diabetes.csv"
])

try:
    df = pd.read_csv(dataset)
    st.success(f"{dataset} loaded successfully ✅")
    st.dataframe(df.head())

except Exception as e:
    st.error(e)
=======
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_client(model, dataloader, epochs=5, lr=0.01, apply_mitigation=False, sensitive_attr=None, y_train=None):
    """Trains a model on client edge node data."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='none' if apply_mitigation else 'mean')

    # If applying bias mitigation (adaptive re-weighting)
    sample_weights = None
    if apply_mitigation and sensitive_attr is not None and y_train is not None:
        sample_weights = torch.ones(len(y_train), dtype=torch.float32)
        priv_mask = (sensitive_attr == 1)
        unpriv_mask = (sensitive_attr == 0)
        pos_mask = (y_train == 1)
        
        # Boost unprivileged positives
        sample_weights[unpriv_mask & pos_mask] *= 1.5
        # Penalize privileged positives
        sample_weights[priv_mask & pos_mask] *= 0.8

    for epoch in range(epochs):
        for batch_idx, (data, target, idx) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            
            if apply_mitigation and sample_weights is not None:
                weights = sample_weights[idx]
                loss = criterion(output, target.view_as(output))
                loss = (loss * weights.view_as(loss)).mean()
            else:
                loss = criterion(output, target.view_as(output))
                
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def federated_averaging(global_model, client_weights):
    """Aggregates local edge models using Secure FedAvg."""
    avg_weights = copy.deepcopy(client_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(client_weights))
    
    global_model.load_state_dict(avg_weights)
    return global_model

def evaluate_model(model, X_test, y_test, sensitive_attr_test):
    """Evaluates Accuracy, Demographic Parity, and Equal Opportunity."""
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test.values, dtype=torch.float32))
        preds = (outputs > 0.5).int().flatten().numpy()
        y_true = y_test.values
        
        accuracy = np.mean(preds == y_true)
        
        # Fairness: Demographic Parity Difference
        priv_idx = (sensitive_attr_test == 1)
        unpriv_idx = (sensitive_attr_test == 0)
        
        priv_pred_pos = np.mean(preds[priv_idx] == 1) if np.sum(priv_idx) > 0 else 0
        unpriv_pred_pos = np.mean(preds[unpriv_idx] == 1) if np.sum(unpriv_idx) > 0 else 0
        dpd = abs(priv_pred_pos - unpriv_pred_pos)
        
        # Fairness: Equal Opportunity Difference
        priv_true_pos_idx = (sensitive_attr_test == 1) & (y_true == 1)
        unpriv_true_pos_idx = (sensitive_attr_test == 0) & (y_true == 1)
        
        priv_tpr = np.mean(preds[priv_true_pos_idx] == 1) if np.sum(priv_true_pos_idx) > 0 else 0
        unpriv_tpr = np.mean(preds[unpriv_true_pos_idx] == 1) if np.sum(unpriv_true_pos_idx) > 0 else 0
        eod = abs(priv_tpr - unpriv_tpr)

    return {"accuracy": accuracy, "dpd": dpd, "eod": eod}
>>>>>>> 2f67b5a5ccbe32c514173a6bcafecf3d78a8e6e6
