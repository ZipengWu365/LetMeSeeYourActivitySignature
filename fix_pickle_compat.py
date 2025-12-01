#!/usr/bin/env python
"""
Fix pickle compatibility issue with numpy._core -> numpy.core
Re-save X_feats.pkl to be compatible with current numpy version.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

class NumpyUnpickler(pickle.Unpickler):
    """Custom unpickler that fixes numpy._core -> numpy.core mapping"""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

def main():
    pkl_path = Path('prepared_data/X_feats.pkl')
    backup_path = Path('prepared_data/X_feats_backup.pkl')
    
    print(f"Loading {pkl_path} with compatibility fix...")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = NumpyUnpickler(f).load()
        
        print(f"  Type: {type(data)}")
        if hasattr(data, 'shape'):
            print(f"  Shape: {data.shape}")
        elif hasattr(data, '__len__'):
            print(f"  Length: {len(data)}")
        
        # Backup original
        import shutil
        if not backup_path.exists():
            shutil.copy(pkl_path, backup_path)
            print(f"  Backed up to {backup_path}")
        
        # Re-save with current numpy/pandas
        if isinstance(data, pd.DataFrame):
            data.to_pickle(pkl_path)
        else:
            # Convert to DataFrame if it's numpy array
            df = pd.DataFrame(data)
            df.to_pickle(pkl_path)
        
        print(f"  Re-saved {pkl_path} with current numpy version")
        
        # Verify
        test = pd.read_pickle(pkl_path)
        print(f"  Verification: OK, shape={test.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
