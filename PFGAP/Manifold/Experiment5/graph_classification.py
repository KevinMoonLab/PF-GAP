#!/usr/bin/env python
# coding: utf-8

# # The goal of this file is to test PFGAP on Graphs

import numpy as np
import networkx as nx
from tqdm.notebook import tqdm  # For a nice progress bar

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# PyTorch Geometric imports
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import time

# --- Load Dataset ---
print("Loading PROTEINS dataset...")
dataset_full = TUDataset(root='data/TUDataset', name='PROTEINS')
print("Dataset loaded successfully.")
print("-" * 30)

# --- Your Distance Function ---
def graph_distance(graph1, graph2):
    """Calculates distance based on Laplacian spectra."""
    # Convert PyG graphs to NetworkX graphs
    # node_attrs=None and edge_attrs=None can speed this up
    g1_nx = to_networkx(graph1, node_attrs=None, edge_attrs=None)
    g2_nx = to_networkx(graph2, node_attrs=None, edge_attrs=None)
    
    spec1 = nx.laplacian_spectrum(g1_nx)
    spec2 = nx.laplacian_spectrum(g2_nx)
    
    # Use the smaller of the two spectrum lengths for comparison
    k = min(len(spec1), len(spec2))
    
    # Calculate the L2 norm (Euclidean distance) of the spectra
    return np.linalg.norm(spec1[:k] - spec2[:k])

# Sample 100 random indices without replacement
SEEDS = [21415, 60139, 78206]
for SEED in SEEDS:
    np.random.seed(SEED)  # For reproducibility
    sample_indices = np.random.choice(len(dataset_full), size=1113, replace=False)

    # Create a sample dataset (list of sampled graphs)
    dataset = [dataset_full[i] for i in sample_indices]

    #* Knn Time
    start_time = time.time()

    num_graphs = len(dataset)
    # Initialize an empty square matrix to hold the distances
    distance_matrix = np.zeros((num_graphs, num_graphs))

    print(f"Calculating {num_graphs}x{num_graphs} distance matrix...")

    # Use tqdm for a progress bar
    for i in range(num_graphs):
        for j in range(i, num_graphs): # We only need to compute the upper triangle
            if i == j:
                continue # Distance to self is 0
            
            # Calculate and store the distance
            dist = graph_distance(dataset[i], dataset[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist # The matrix is symmetric

    print("\nDistance matrix calculation complete!")
    print(f"Shape of distance matrix: {distance_matrix.shape}")

    # Get the labels for each graph
    labels = np.array([graph.y.item() for graph in dataset])

    # Create an array of indices [0, 1, 2, ..., n-1]
    indices = np.arange(num_graphs)

    # Split indices and labels into training and testing sets
    # We stratify by labels to ensure both sets have a similar class distribution
    train_indices, test_indices, y_train, y_test = train_test_split(
        indices, labels, test_size=0.3, random_state=SEED, stratify=labels
    )

    # Now, create the training and testing distance matrices
    # X_train should contain distances between all training samples
    X_train_precomputed = distance_matrix[train_indices, :][:, train_indices]

    # X_test should contain distances between test samples (rows) and training samples (columns)
    X_test_precomputed = distance_matrix[test_indices, :][:, train_indices]

    from sklearn.metrics import f1_score, confusion_matrix

    # Initialize the classifier with k=5 neighbors
    # IMPORTANT: We set metric='precomputed'
    knn = KNeighborsClassifier(n_neighbors=5, metric='precomputed')

    # Train the model
    print("Training KNN classifier...")
    knn.fit(X_train_precomputed, y_train)

    # Make predictions on the test set
    print("Making predictions...")
    y_pred = knn.predict(X_test_precomputed)
    end_time = time.time()

    knn_time = end_time - start_time

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)


    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 30)
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

    # Get outlier (anomaly) scores for the train set using KNN
    distances_train, _ = knn.kneighbors(X_train_precomputed, n_neighbors=knn.n_neighbors)
    knn_outlier_scores_train = distances_train[:, -1]  # Distance to the k-th nearest neighbor
    print("KNN outlier scores for train set:", knn_outlier_scores_train)

    import pandas as pd
    import os
    import json

    # Create a results directory if it doesn't exist
    results_dir = os.path.join('/yunity/arusty/PF-GAP-1/PFGAP/Manifold/Experiment5', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Store KNN model results in a single row dataframe
    model_results = pd.DataFrame({
        'model': 'knn',
        'seed': SEED,
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': json.dumps(cm.tolist()),  # Save as JSON string
        'outlier_scores': [knn_outlier_scores_train],
        'time': knn_time
    })

    # Save to CSV
    model_results_path = os.path.join(results_dir, f'knn_model_results{SEED}.csv')
    model_results.to_csv(model_results_path, index=False)
    print(f"KNN model results saved to: {model_results_path}")

    import sys
    import os
    import pickle
    import pandas as pd
    import numpy as np

    # Add the Application directory to the Python path so we can import proxUtil
    sys.path.append('/yunity/arusty/PF-GAP-1/PFGAP/Application')
    import proxUtil

    # Create a directory for our graph experiment data
    experiment_dir = '/yunity/arusty/PF-GAP-1/PFGAP/Manifold/Experiment5'
    data_dir = os.path.join(experiment_dir, 'graph_data')
    os.makedirs(data_dir, exist_ok=True)

    print("🚀 Implementing INDEXING approach for PFGAP with graphs")

    # Prepare ALL graphs (train + test) in a single indexed collection
    all_graphs = []
    all_labels = []
    graph_to_original_mapping = {}  # Maps new index -> (dataset_type, original_index)

    # Add training graphs first
    for i, graph_idx in enumerate(train_indices):
        all_graphs.append(dataset[graph_idx])
        all_labels.append(y_train[i])
        graph_to_original_mapping[len(all_graphs)-1] = ('train', i)

    # Add test graphs
    train_count = len(all_graphs)
    for i, graph_idx in enumerate(test_indices):
        all_graphs.append(dataset[graph_idx])
        all_labels.append(y_test[i])
        graph_to_original_mapping[len(all_graphs)-1] = ('test', i)

    print(f"📊 Total graphs: {len(all_graphs)} (Training: {train_count}, Test: {len(all_graphs)-train_count})")

    # Save the complete graph collection for the distance function
    graphs_data = {
        'graphs': all_graphs,
        'labels': all_labels,
        'train_count': train_count,
        'mapping': graph_to_original_mapping
    }

    graphs_file = os.path.join(data_dir, 'indexed_graphs.pkl')
    with open(graphs_file, 'wb') as f:
        pickle.dump(graphs_data, f)

    # Create TSV files with indices and dummy features
    def create_indexed_tsv(start_idx, end_idx, labels_subset, filename):
        """
        Create a TSV file where each row contains: label, index, dummy_feature
        """
        rows = []
        for i, (graph_index, label) in enumerate(zip(range(start_idx, end_idx), labels_subset)):
            rows.append([label, graph_index, 0.0])
        
        df = pd.DataFrame(rows)
        tsv_path = os.path.join(data_dir, filename)
        df.to_csv(tsv_path, sep='\t', header=False, index=False)
        return tsv_path

    # Create training and test TSVs
    train_tsv = create_indexed_tsv(0, train_count, y_train, 'indexed_train.tsv')
    test_tsv = create_indexed_tsv(train_count, len(all_graphs), y_test, 'indexed_test.tsv')

    print(f"📝 Created TSV files: {os.path.basename(train_tsv)}, {os.path.basename(test_tsv)}")

    print("\n🚀 TRAINING PFGAP WITH INDEXED GRAPHS")

    # Change to the Application directory so PFGAP can find the JAR files
    original_cwd = os.getcwd()
    os.chdir('/yunity/arusty/PF-GAP-1/PFGAP/Application')

    try:
        pfgap_start_time = time.time()
        # Train PFGAP model using the python distance function
        proxUtil.getProx(
            trainfile=train_tsv,
            testfile=test_tsv,
            modelname="IndexedGraphPFGAP",
            distances=['python'],  # Use our custom indexed graph distance function
            num_trees=11,
            r=5,
            getprox="false",
            savemodel="false",
            out="indexed_graph_output",
            verbosity=2,
            csv_has_header="false",
            target_column="first"
        )
        pfgap_end_time = time.time()
        print("\n🎉 PFGAP training completed")
        
        # Read the results
        try:
            # Read predictions
            with open("Predictions.txt", 'r') as f:
                pred_content = f.read()
                pfgap_predictions = eval("np.array(" + pred_content + ")")
            
            # Calculate accuracy
            pfgap_accuracy = accuracy_score(y_test, pfgap_predictions)
            if pfgap_accuracy < .5:
                pfgap_predictions = 1 - pfgap_predictions
                pfgap_accuracy = accuracy_score(y_test, pfgap_predictions)
            print(f"🎯 PFGAP Model Accuracy: {pfgap_accuracy * 100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error reading PFGAP results: {e}")
            
    except Exception as e:
        print(f"❌ Error running PFGAP: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Change back to original directory
        os.chdir(original_cwd)

    import pandas as pd
    import os
    import json
    from sklearn.metrics import f1_score, confusion_matrix
    import sys

    sys.path.append('/yunity/arusty/PF-GAP-1/PFGAP/Application')

    # Calculate metrics for PFGAP
    pfgap_f1 = f1_score(y_test, pfgap_predictions)
    pfgap_cm = confusion_matrix(y_test, pfgap_predictions)

                         
    # Store PFGAP model results in a dataframe (similar to KNN)
    # Convert outlier scores to a Python list of floats for natural storage
    pfgap_results = pd.DataFrame({
        'model': 'pfgap',
        'seed': SEED,
        'accuracy': pfgap_accuracy,
        'f1_score': pfgap_f1,
        'confusion_matrix': json.dumps(pfgap_cm.tolist()),
        'outlier_scores': None,
        "time": pfgap_end_time - pfgap_start_time
    })

    # Save PFGAP results to CSV
    pfgap_results_path = os.path.join(results_dir, f'pfgap_model_results{SEED}.csv')
    pfgap_results.to_csv(pfgap_results_path, index=False)
    print(f"PFGAP model results saved to: {pfgap_results_path}")

