import subprocess
import sys

# test_pfgap_single_seed.py
from test_pipeline import run_single_test, SEEDS

def test_pfgap_single_seed():
    seed = SEEDS[0] if SEEDS else 42
    result = run_single_test(seed)
    print("PFGAP Results for seed", seed)
    print(result.get('pfgap_euclidean', 'No PFGAP results found'))

if __name__ == "__main__":
    test_pfgap_single_seed()
