# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:07:08 2026

@author: N31937
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import pyamg
from scipy.io import mmread, mminfo
from scipy.sparse.linalg import svds, lobpcg, LinearOperator
from scipy.sparse.csgraph import maximum_bipartite_matching
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def get_sorted_matrix_files(matrix_dir):
    file_metadata = []
    for f in matrix_dir.rglob('*.mtx'):
        if f.is_file():
            try:
                # mminfo reads just the header: (rows, cols, entries, format, field, symmetry)
                info = mminfo(f)
                nnz = info[2] if len(info) > 2 else (info[0] * info[1])
                file_metadata.append((f.stem, f, nnz))
            except Exception:
                file_metadata.append((f.stem, f, float('inf')))
    # Sort smallest NNZ to largest NNZ
    file_metadata.sort(key=lambda x: x[2])
    return file_metadata

def calculate_rank_metrics(csv_path, matrix_dir):
    MAX_RANK_DIMENSION = 5000000
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return
    sorted_files = get_sorted_matrix_files(matrix_dir)
    print(f"Found {len(sorted_files)} .mtx files. Checking for rank gaps...\n")
    modifications = 0
    target_columns = [
        'Structural Rank', 'Structural Rank Full', 'Rank',
        'Full Numerical Rank?', 'sprank(A)-rank(A)',
        'Null Space Dimension', 'Num Dmperm Blocks'
    ]
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan
    if 'Name' in df.columns:
        df['NameClean'] = df['Name'].astype(str).str.strip()
    else:
        print("Could not find Name in CSV")
        return
    for name, file_path, nnz in sorted_files:
        cleanName = name.strip()
        idx = df.index[df['NameClean'].astype(str) == cleanName]
        if idx.empty:
            continue
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True).replace('NaN', np.nan, inplace=True)
        idx = idx[0]
        needs_calc = any(pd.isna(df.at[idx, col]) for col in target_columns)
        if not needs_calc:
            continue
        rows, cols = df.at[idx, 'Num Rows'], df.at[idx, 'Num Cols']
        if pd.isna(rows) or pd.isna(cols):
            continue
        can_compute_numerical = (rows <= MAX_RANK_DIMENSION and cols <= MAX_RANK_DIMENSION)
        calcs = {}
        struct_rank = df.at[idx, 'Structural Rank']
        struct_rank_full = df.at[idx, 'Structural Rank Full']
        try:
            needs_matrix = pd.isna(struct_rank) or (can_compute_numerical and pd.isna(df.at[idx, 'Rank']))
            if needs_matrix:
                A = mmread(file_path).tocsr()
                print(f"Processing ranks for {name} ({int(rows)}x{int(cols)})...")
            # --- Structural Rank ---
            if pd.isna(struct_rank):
                matching = maximum_bipartite_matching(A)
                struct_rank = int((matching >= 0).sum())
                struct_rank_full = 1 if struct_rank == min(rows, cols) else 0
                calcs['Structural Rank'] = struct_rank
                calcs['Structural Rank Full'] = struct_rank_full
            # --- Dmperm Blocks ---
            if pd.isna(df.at[idx, 'Num Dmperm Blocks']) and struct_rank_full == 1:
                scc_val = df.at[idx, 'Strongly Connect Components']
                if pd.notna(scc_val):
                    calcs['Num Dmperm Blocks'] = scc_val
            # --- Exact Numerical Rank & Derived Metrics ---
            if pd.isna(df.at[idx, 'Rank']):
                # Safe limit for dense arrays (12,000 x 12,000 = ~1.1 GB RAM)
                DENSE_RAM_LIMIT = 300000
                num_rank = None
                if rows <= DENSE_RAM_LIMIT and cols <= DENSE_RAM_LIMIT:
                    # Matrix is small enough to safely convert to dense and calculate exact integer rank
                    print(" -> Matrix is within safe RAM limits. Calculating exact dense numerical rank...")
                    num_rank = np.linalg.matrix_rank(A.toarray())
                elif can_compute_numerical:
                    # Matrix is too big for dense array. Use Sparse Preconditioned SVD shortcut.
                    print(f" -> Matrix too large for exact dense rank (>{DENSE_RAM_LIMIT}). Attempting sparse Full-Rank verification...")
                    try:
                        # Ensure float type for solver
                        A_float = A.astype(float)
                        H = A_float.T @ A_float
                        n_H = H.shape[0]
                        class StrictJacobiPreconditioner(LinearOperator):
                            def __init__(self, diag, size, dtype):
                                self.diag = diag
                                self.shape = (size, size)
                                self.dtype = dtype
                            def _matvec(self, x):
                                return x / self.diag
                            def _matmat(self, X):
                                return X / self.diag[:, None]
                        shift = 1e-5
                        H_shifted = H + shift * sp.eye(n_H)
                        # Use Jacobi directly here as A^T A can be too dense for AMG to set up quickly
                        diag_H = H_shifted.diagonal()
                        diag_H[diag_H < 1e-12] = 1.0
                        M = StrictJacobiPreconditioner(diag_H, n_H, H_shifted.dtype)
                        X = np.random.rand(n_H, 1)
                        # Calculate smallest eigenvalue
                        eigenvalues, _ = lobpcg(H, X, M=M, largest=False, tol=1e-5, maxiter=1000)
                        min_sv = np.sqrt(max(0.0, float(eigenvalues[0])))
                        if min_sv > 1e-11:
                            print(f" [Verified] Minimum SV is {min_sv:.2e} (>0). Matrix is mathematically Full Rank.")
                            num_rank = min(rows, cols)
                        else:
                            print(" [Deficient] Minimum SV is ~0. Matrix is rank-deficient. Skipping exact integer rank to prevent RAM crash.")
                    except Exception as lobpcg_err:
                        print(f" [Error] Sparse rank verification failed: {lobpcg_err}")

                # If we successfully found the rank (either exactly, or verified as full rank)
                if num_rank is not None:
                    num_rank_full = 1 if num_rank == min(rows, cols) else 0
                    rank_diff = struct_rank - num_rank
                    null_space = cols - num_rank
                    calcs.update({
                        'Rank': num_rank,
                        'Full Numerical Rank?': num_rank_full,
                        'sprank(A)-rank(A)': rank_diff,
                        'Null Space Dimension': null_space
                    })
            # --- Update CSV In-Place ---
            for col, val in calcs.items():
                if pd.isna(df.at[idx, col]):
                    df.at[idx, col] = val
                    modifications += 1
            if modifications >= 1:
                df.to_csv(csv_path, index=False)
                print(f" -> Processed {name} successfully.")
        except Exception as e:
            print(f" Error processing {name}: {e}")
    print(f"Filled {modifications} rank-related metric gaps.\n")

def calculate_advanced_metrics(csv_path, matrix_dir):
    MAX_MATRIX_DIMENSION = 5000000
    MAX_ARPACK_ITERATIONS = 2000
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return
    sorted_files = get_sorted_matrix_files(matrix_dir)
    print(f"Found {len(sorted_files)} .mtx files. Starting advanced preconditioned calculations...\n")
    modifications = 0
    target_cols = ['Matrix Norm', 'Minimum Singular Value', 'Condition Number']
    for name, file_path, nnz in sorted_files:
        idx = df.index[df['Name'].astype(str) == name]
        if idx.empty:
            continue
        idx = idx[0]
        needs_calc = any(pd.isna(df.at[idx, col]) for col in target_cols if col in df.columns)
        if not needs_calc:
            continue
        rows, cols = df.at[idx, 'Num Rows'], df.at[idx, 'Num Cols']
        if pd.isna(rows) or pd.isna(cols) or rows > MAX_MATRIX_DIMENSION or cols > MAX_MATRIX_DIMENSION:
            continue
        print(f"Loading and calculating {name} ({int(rows)}x{int(cols)})...")
        try:
            A = mmread(file_path).tocsc().astype(float)
            # 1. Max Singular Value
            print(" -> Computing Max Singular Value (Norm)...")
            _, max_sv_array, _ = svds(A, k=1, which='LM', maxiter=MAX_ARPACK_ITERATIONS)
            max_sv = float(max_sv_array[0])
            # 2. Min Singular Value via Preconditioned LOBPCG
            print(" -> Computing Min Singular Value via A^T A Eigenvalue Transformation...")
            H = A.T @ A
            n_H = H.shape[0]
            class StrictAMGPreconditioner(LinearOperator):
                def __init__(self, ml_solver, size, dtype):
                    self.ml_solver = ml_solver
                    self.shape = (size, size)
                    self.dtype = dtype
                def _matvec(self, x):
                    return self.ml_solver.solve(x, tol=1e-5)
                def _matmat(self, X):
                    return self.ml_solver.solve(X, tol=1e-5)
            class StrictJacobiPreconditioner(LinearOperator):
                def __init__(self, diag, size, dtype):
                    self.diag = diag
                    self.shape = (size, size)
                    self.dtype = dtype
                def _matvec(self, x):
                    return x / self.diag
                def _matmat(self, X):
                    return X / self.diag[:, None]

            shift = 1e-5
            H_shifted = H + shift * sp.eye(n_H)
            print(" -> Building Preconditioner...")
            try:
                diag_H = H_shifted.diagonal()
                diag_H[diag_H < 1e-12] = 1.0
                M = StrictJacobiPreconditioner(diag_H, n_H, H_shifted.dtype)
                
            except Exception as amg_err:
                print(f" [AMG failed: {amg_err}. Falling back to zero-memory Jacobi]")

                ml = pyamg.smoothed_aggregation_solver(H_shifted, max_coarse=100)
                M = StrictAMGPreconditioner(ml, n_H, H_shifted.dtype)
                print(" [AMG Preconditioner successfully built]")
            X = np.random.rand(n_H, 1)
            eigenvalues, _ = lobpcg(H, X, M=M, largest=False, tol=1e-5, maxiter=MAX_ARPACK_ITERATIONS)
            min_eig = max(0.0, float(eigenvalues[0]))
            min_sv = np.sqrt(min_eig)
            # 3. Condition Number
            cond_num = max_sv / min_sv if min_sv > 1e-15 else np.inf
            calcs = {
                'Matrix Norm': max_sv,
                'Minimum Singular Value': min_sv,
                'Condition Number': cond_num
            }
            for col, val in calcs.items():
                if col in df.columns and pd.isna(df.at[idx, col]):
                    df.at[idx, col] = val
                    modifications += 1
                    print(f" Filled gap for {col}: {val:.4e}")
        except Exception as e:
            print(f" Error calculating {name}: {e}")
        # Save in-place incrementally
        df.to_csv(csv_path, index=False)
    print(f"Filled {modifications} complex metric gaps.")

if __name__ == "__main__":
    base_directory = Path('.')
    csv_file = base_directory / 'testMatrices.csv'
    matrix_folder = base_directory / 'Test Files'
    if not matrix_folder.exists():
        print(f"Error: Directory '{matrix_folder.name}' does not exist.")
    else:
        # Run pipeline sequentially, updating the same CSV file each time
        calculate_rank_metrics(csv_file, matrix_folder)
        calculate_advanced_metrics(csv_file, matrix_folder)