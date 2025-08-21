# PFGAP Graph Distance using Indexing Approach
# This file implements a graph distance function that uses indices to look up graphs
# instead of serializing the graph data directly in the TSV files.

import pickle
import os
import sys

# Global variables to store loaded graph data (loaded once when module is imported)
GRAPHS_DATA = None
GRAPHS_LOADED = False

def load_graphs_data():
    """
    Load the graph data once when the module is first imported.
    This avoids repeatedly loading the same data for each distance calculation.
    """
    global GRAPHS_DATA, GRAPHS_LOADED
    
    if GRAPHS_LOADED:
        return GRAPHS_DATA
    
    try:
        # Path to the indexed graphs file
        graphs_file = '/yunity/arusty/PF-GAP-1/PFGAP/Manifold/Experiment5/graph_data/indexed_graphs.pkl'
        
        if not os.path.exists(graphs_file):
            print(f"Error: Indexed graphs file not found: {graphs_file}")
            return None
            
        with open(graphs_file, 'rb') as f:
            GRAPHS_DATA = pickle.load(f)
            
        GRAPHS_LOADED = True
        print(f"✅ Loaded {len(GRAPHS_DATA['graphs'])} graphs for distance calculations")
        return GRAPHS_DATA
        
    except Exception as e:
        print(f"Error loading graphs data: {e}")
        return None

def graph_distance_laplacian(graph1, graph2):
    """
    Calculate distance between two graphs based on Laplacian spectra.
    """
    try:
        # Import required libraries
        import torch
        import networkx as nx
        import numpy as np
        from torch_geometric.utils import to_networkx
        
        # Convert PyG graphs to NetworkX graphs
        g1_nx = to_networkx(graph1, node_attrs=None, edge_attrs=None)
        g2_nx = to_networkx(graph2, node_attrs=None, edge_attrs=None)
        
        # Calculate Laplacian spectra
        spec1 = nx.laplacian_spectrum(g1_nx)
        spec2 = nx.laplacian_spectrum(g2_nx)
        
        # Use the smaller of the two spectrum lengths for comparison
        k = min(len(spec1), len(spec2))
        
        # Calculate the L2 norm (Euclidean distance) of the spectra
        distance = float(np.linalg.norm(spec1[:k] - spec2[:k]))
        return distance
        
    except ImportError as e:
        print(f"Required packages not available: {e}")
        return 1.0
    except Exception as e:
        print(f"Error calculating graph distance: {e}")
        return 1.0

def Distance(t1, t2):
    """
    Distance function for PFGAP using the indexing approach.
    
    Args:
        t1, t2: Arrays from Java, where t1[0] is the graph index for the first graph
                and t2[0] is the graph index for the second graph.
                
    Returns:
        float: Distance between the two graphs
    """
    try:
        # Load graph data if not already loaded
        graphs_data = load_graphs_data()
        if graphs_data is None:
            return 1.0
            
        # Extract graph indices from the input arrays
        # Java will pass arrays like [index] where index is the graph position
        if len(t1) < 1 or len(t2) < 1:
            print(f"Error: Input arrays too short: {len(t1)}, {len(t2)}")
            return 1.0
            
        try:
            # The graph index should be the first (and only) element in the array
            index1 = int(t1[0])
            index2 = int(t2[0])
        except (ValueError, IndexError) as e:
            print(f"Error extracting graph indices: {e}")
            print(f"t1: {t1}, t2: {t2}")
            return 1.0
            
        # Validate indices
        if index1 < 0 or index1 >= len(graphs_data['graphs']):
            print(f"Error: Index1 {index1} out of range [0, {len(graphs_data['graphs'])})")
            return 1.0
            
        if index2 < 0 or index2 >= len(graphs_data['graphs']):
            print(f"Error: Index2 {index2} out of range [0, {len(graphs_data['graphs'])})")
            return 1.0
            
        # Get the actual graphs
        graph1 = graphs_data['graphs'][index1]
        graph2 = graphs_data['graphs'][index2]
        
        # Calculate and return the distance
        return graph_distance_laplacian(graph1, graph2)
        
    except Exception as e:
        print(f"Error in Distance function: {e}")
        import traceback
        traceback.print_exc()
        return 1.0