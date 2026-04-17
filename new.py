import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
from scipy.sparse.linalg import norm
from pathlib import Path

def update_test_matrices(csv_path='testMatrices.csv', matrix_dir_path='./matrices'):
    # 1. Load the dataset
    df = pd.read_csv(csv_path)
    
    # 2. Initialize missing columns if they don't exist
    new_cols = [
        'Skew-Symmetric Frobenius Norm',
        'Diagonal Dominance Min Ratio',
        'Diagonal Dominance Mean Ratio',
        'Diagonal Dominance Max Ratio'
    ]
    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    # 3. Scan for .mtx files
    matrix_dir = Path(matrix_dir_path)
    # Hunts down every .mtx file in the directory (and subdirectories)
    mtx_files = {f.stem: f for f in matrix_dir.rglob('*.mtx') if f.is_file()}
    print(f"Found {len(mtx_files)} matrix files in {matrix_dir_path}\n")

    # 4. Process each matrix
    for idx, row in df.iterrows():
        matrix_name = str(row['Name'])
        
        if matrix_name not in mtx_files:
            print(f"Warning: {matrix_name}.mtx not found. Skipping...")
            continue
            
        try:
            # Load and convert to CSR for fast row operations
            A = mmread(mtx_files[matrix_name]).tocsr()
            n_rows, n_cols = A.shape
            
            # --- 1. Skew-Symmetric Frobenius Norm ---
            try:
                AS = 0.5 * (A - A.T)
                skew_norm = norm(AS, ord='fro')
                df.at[idx, 'Skew-Symmetric Frobenius Norm'] = skew_norm
            except Exception as e:
                print(f"Error computing Skew-Symmetric Norm for {matrix_name}: {e}")
                
            # --- 2. Diagonal Dominance Metrics ---
            if n_rows == n_cols:  # Must be a square matrix
                try:
                    diag = np.real(A.diagonal())
                    
                    # Calculate absolute off-diagonal sums using fast CSR math
                    abs_A = A.copy()
                    abs_A.data = np.abs(abs_A.data)
                    row_sums = np.array(abs_A.sum(axis=1)).flatten()
                    offdiag_sums = row_sums - np.abs(diag)
                    
                    # Vectorized ratio calculation (replicated exactly from project.ipynb)
                    ratios = np.divide(
                        np.abs(diag), 
                        offdiag_sums, 
                        out=np.zeros_like(diag, dtype=float), 
                        where=offdiag_sums != 0
                    )
                    
                    # Handle isolated diagonal elements
                    ratios[offdiag_sums == 0] = np.inf
                    finite_ratios = ratios[np.isfinite(ratios)]
                    
                    df.at[idx, 'Diagonal Dominance Min Ratio'] = float(np.min(finite_ratios)) if len(finite_ratios) > 0 else 0.0
                    df.at[idx, 'Diagonal Dominance Mean Ratio'] = float(np.mean(finite_ratios)) if len(finite_ratios) > 0 else 0.0
                    df.at[idx, 'Diagonal Dominance Max Ratio'] = float(np.max(finite_ratios)) if len(finite_ratios) > 0 else np.inf
                    
                except Exception as e:
                    print(f"Error computing Diagonal Dominance for {matrix_name}: {e}")
            else:
                # If rectangular, dominance metrics don't apply
                df.at[idx, 'Diagonal Dominance Min Ratio'] = np.nan
                df.at[idx, 'Diagonal Dominance Mean Ratio'] = np.nan
                df.at[idx, 'Diagonal Dominance Max Ratio'] = np.nan
                
            print(f"Successfully updated: {matrix_name}")
                
        except Exception as e:
            print(f"Error loading {matrix_name}: {e}")
            
    # 5. Save the updated dataframe
    output_path = 'testMatrices_updated.csv'
    df.to_csv(output_path, index=False)
    print(f"\nFinished! Updated test set saved to: {output_path}")

# ==========================================
# EXECUTION
# ==========================================
# Change the matrix_dir_path to the folder where your .mtx files actually live!
update_test_matrices(
    csv_path='testMatrices.csv', 
    matrix_dir_path=r'X:\.win_desktop\GIT\DataAnalytics')