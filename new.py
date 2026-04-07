#========================================================================
# Dual Proxy Generator: Topology (Laplacian) & Algebra (Signed Bipartite)
#========================================================================
import pandas as pd
import numpy as np
import json
import pyamg
from scipy.io import mmread
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import lobpcg, LinearOperator
from pathlib import Path
import warnings
import gc
import os
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian
warnings.filterwarnings("ignore", category=UserWarning)

def add_custom_matrix_metrics():
    base_dir = Path('.')
    csv_file_path = base_dir / 'transforms.csv'
    matrix_dir = base_dir / 'Matrix Files'
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file_path}")
        return
    df.reset_index(drop=True, inplace=True)
    if 'Name' not in df.columns:
        df.rename(columns={df.columns[2]: 'Name'}, inplace=True)
    df['Name'] = df['Name'].astype(str).str.replace('_laplacian', '').str.strip()
    base_names = df['Name'].unique()
    
    custom_columns = [
        'RCM Bandwidth', 'Directional Mean Bias', 'Signed Frobenius Ratio', 'Gershgorin Discs',
        'Strictly Diagonally Dominant Row Fraction', 'Fiedler Value',
        'Brauer Max Product', 'Brauer Mean Product', 'Brauer Mean Product (Top)',
        'Brauer Min Product', 'Brauer Max Center Distance', 'Brauer Mean Center Distance'
    ]
    for col in custom_columns:
        if col not in df.columns:
            df[col] = np.nan
    for text_col in ['Gershgorin Discs', 'Brauer Unions']:
        if text_col in df.columns:
            df[text_col] = df[text_col].astype('object')
    modifications = 0
    
    files = [f for f in os.listdir(matrix_dir) if f.endswith('.mtx')]
    for filename in files:
        name=filename.replace('.mtx', '')
        filepath = os.path.join(matrix_dir, filename)
    
        matches = df.index[df['Name'] == name]
        if len(matches) == 0:
            continue
        idx = matches[0]
        needs_calc = any(pd.isna(df.at[idx, col]) for col in custom_columns)
        if not needs_calc:
            continue
        print(f"Loading and calculating custom metrics for {name}...")
        try:
            # Load proxies into memory
            A = mmread(filepath).tocsr()
            A_alg = sp.bmat([[None, A], [A.T, None]], format='csr')
            A_top = laplacian(sp.bmat([[None, np.abs(A)], [np.abs(A).T, None]], format='csr'))
            n = A_alg.shape[0]
            calcs = {}
            # --- Positivity Metrics (Algebraic) ---
            if pd.isna(df.at[idx, 'Signed Frobenius Ratio']):
                try:
                    data = np.nan_to_num(A_alg.data, nan=0.0, posinf=1e150, neginf=-1e150)
                    pos_data = data[data.real > 0]
                    neg_data = data[data.real < 0]
                    if len(pos_data) > 0 or len(neg_data) > 0:
                        mean_p = np.mean(pos_data) if len(pos_data) > 0 else 0.0
                        mean_n = np.mean(neg_data) if len(neg_data) > 0 else 0.0
                        mag_p, mag_n = np.abs(mean_p), np.abs(mean_n)
                        if mag_p > mag_n:
                            calcs['Directional Mean Bias'] = (mag_p / mag_n) if mag_n != 0 else np.inf
                        else:
                            calcs['Directional Mean Bias'] = -(mag_p / mag_n) if mag_n != 0 else -np.inf
                    else:
                        calcs['Directional Mean Bias'] = 0.0
                    if len(pos_data) > 0 or len(neg_data) > 0:
                        norm_p = np.sqrt(np.sum(np.nan_to_num(np.abs(pos_data))**2, nan=0.0, posinf=1e150, neginf=-1e150)) if len(pos_data) > 0 else 0.0
                        norm_n = np.sqrt(np.sum((np.abs(neg_data))**2)) if len(neg_data) > 0 else 0.0
                        if norm_n > norm_p:
                            calcs['Signed Frobenius Ratio'] = -(norm_n / norm_p) if norm_p != 0 else -np.inf
                        else:
                            calcs['Signed Frobenius Ratio'] = (norm_p / norm_n) if norm_n != 0 else np.inf
                    else:
                        calcs['Signed Frobenius Ratio'] = 0.0
                except Exception as e:
                    print(f" -> Error calculating Positivity Metrics: {e}")
            # --- RCM Bandwidth (Topological) ---
            if pd.isna(df.at[idx, 'RCM Bandwidth']):
                try:
                    perm = reverse_cuthill_mckee(A_top, symmetric_mode=True)
                    inv_perm = np.argsort(perm)
                    rows, cols = A_top.nonzero()
                    if len(rows) > 0:
                        calcs['RCM Bandwidth'] = int(np.max(np.abs(inv_perm[rows] - inv_perm[cols])))
                    else:
                        calcs['RCM Bandwidth'] = 0
                except Exception as e:
                    print(f" -> Error calculating RCM Bandwidth: {e}")
            # --- Fast Diagonal & Row Operations (Algebraic) ---
            if n == A_alg.shape[1]:
                diag = np.real(A_alg.diagonal())
                abs_A_alg = np.abs(A_alg)
                row_sums_alg = np.array(abs_A_alg.sum(axis=1)).flatten()
                # --- Gershgorin Discs (Algebraic) ---
                if pd.isna(df.at[idx, 'Gershgorin Discs']):
                    try:
                        L = -row_sums_alg
                        R = row_sums_alg
                        intervals = np.column_stack((L, R))
                        intervals = intervals[intervals[:, 0].argsort()]
                        merged_gershgorin = []
                        for interval in intervals:
                            l_val, r_val = float(interval[0]), float(interval[1])
                            if not merged_gershgorin or round(merged_gershgorin[-1][1], 4) < round(l_val, 4):
                                merged_gershgorin.append([l_val, r_val])
                            else:
                                merged_gershgorin[-1][1] = max(merged_gershgorin[-1][1], r_val)
                        formatted_gersh = [[round(u[0], 4), round(u[1], 4)] for u in merged_gershgorin]
                        calcs['Gershgorin Discs'] = json.dumps(formatted_gersh, separators=(',', ':'))
                    except Exception as e:
                        print(f" -> Error calculating Gershgorin: {e}")
                        calcs['Gershgorin Discs'] = '[]'
                # --- Brauer Products (Algebraic) ---
                if pd.isna(df.at[idx, 'Brauer Max Product']):
                    try:
                        # R_i * R_j for all valid edges.
                        rows, cols = A_alg.nonzero()
                        if len(rows) > 0:
                            # Multiply the row sum of node i with the row sum of node j
                            products = row_sums_alg[rows] * row_sums_alg[cols]
                            calcs['Brauer Max Product'] = float(np.max(products))
                            calcs['Brauer Mean Product'] = float(np.mean(products))
                            calcs['Brauer Min Product'] = float(np.min(products))
                            calcs['Brauer Pair Count'] = len(products) // 2
                        else:
                            calcs['Brauer Max Product'] = 0.0
                            calcs['Brauer Mean Product'] = 0.0
                            calcs['Brauer Min Product'] = 0.0
                            calcs['Brauer Pair Count'] = 0
                    except Exception as e:
                        print(f" -> Error calculating Algebraic Brauer Products: {e}")
            if n == A_top.shape[1]:
                degree_top = np.real(A_top.diagonal())
                if pd.isna(df.at[idx, 'Brauer Mean Product (Top)']):
                    try:
                        A_edges_only = A_top.copy()
                        A_edges_only.setdiag(0)
                        A_edges_only.eliminate_zeros()
                        A_edges_only.data = np.ones_like(A_edges_only.data)
                        R_topo = np.array(A_edges_only.sum(axis=1)).flatten()
                        rows_top, cols_top = A_edges_only.nonzero()
                        if len(rows_top) > 0:
                            products_top = R_topo[rows_top] * R_topo[cols_top]
                            calcs['Brauer Mean Product (Top)'] = float(np.mean(products_top))
                        else:
                            calcs['Brauer Mean Product (Top)'] = 0.0
                    except Exception as e:
                        print(f" -> Error calculating Topological Brauer Distances: {e}")   
                # --- Brauer Center Distances (Topological) ---
                if pd.isna(df.at[idx, 'Brauer Max Center Distance']):
                    try:
                        rows, cols = A_top.nonzero()
                        valid_edges = rows != cols
                        u = rows[valid_edges]
                        v = cols[valid_edges]
                        if len(u) > 0:
                            center_distances = np.abs(degree_top[u] - degree_top[v])
                            calcs['Brauer Max Center Distance'] = float(np.max(center_distances))
                            calcs['Brauer Mean Center Distance'] = float(np.mean(center_distances))
                        else:
                            calcs['Brauer Max Center Distance'] = 0.0
                            calcs['Brauer Mean Center Distance'] = 0.0
                        isolated_count = np.sum(degree_top < 1e-12)
                        calcs['Strictly Diagonally Dominant Row Fraction'] = float(isolated_count / n)
                    except Exception as e:
                        print(f" -> Error calculating Topological Brauer Distances: {e}")    
                # --- Fiedler Value (Topological) ---
                if pd.isna(df.at[idx, 'Fiedler Value']):
                    try:
                        A_sym = 0.5 * (A_top + A_top.T).astype(np.float64)
                        if n > 2:
                            A_shifted = A_sym + 1e-5 * sp.eye(n)
                            Y = np.ones((n, 1))
                            degrees = A_sym.diagonal().reshape(n,1)
                            centered = degrees - np.mean(degrees)
                            X = centered + (0.05 * np.random.rand(n,1))
                            try:
                                ml = pyamg.smoothed_aggregation_solver(A_shifted, presmoother='jacobi', postsmoother='jacobi')
                                M = ml.aspreconditioner()
                                eigenvalues, _ = lobpcg(A_sym, X, M=M, Y=Y, largest=False, tol=1e-5, maxiter=1000)
                                calcs['Fiedler Value'] = max(0.0, float(eigenvalues[0]))
                            except Exception as arpack_err:
                                print(f" -> ARPACK failed ({arpack_err}). Skipping...")
                                #calcs['Fiedler Value'] = -1.0
                        else:
                            calcs['Fiedler Value'] = 0.0
                    except Exception as e:
                        print(f" -> Error calculating Fiedler Value: {e}")
                        calcs['Fiedler Value'] = -1.0
            # --- Update the DataFrame and Save ---
            for col, val in calcs.items():
                if isinstance(val, (list, tuple, np.ndarray)):
                    cleaned = str(val).replace('\n', '').replace('\r', '').replace(' ', '')
                elif isinstance(val, str):
                    cleaned = val.replace('\n', '').replace('\r', '')
                elif isinstance(val, (np.integer, int)):
                    cleaned = int(val)
                elif isinstance(val, (np.floating, float)):
                    cleaned = float(val)
                else:
                    cleaned = val
                df.at[idx, col] = cleaned
                modifications += 1
            # Save progress incrementally after each matrix is done
            df.to_csv(csv_file_path, index=False)
            print(f" -> Processed {name} successfully. Progress saved.")
        except Exception as e:
            print(f" Error loading or processing {name}: {e}")
        # Manual Garbage Collection to prevent memory overflow
        finally:
            for v in ['A', 'A_sym', 'A_shifted', 'A_alg', 'A_top', 'abs_A_alg', 'R_i_j_matrix', 'R_j_i_matrix', 'discriminant']:
                if v in locals():
                    del locals()[v]
            gc.collect()
    print(f"\nFinished processing. Added/Updated {modifications} metric values.")

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    input_folder = r"C:\Users\N31937\OneDrive - NGC\Desktop\Coding\DataAnalytics"
    output_folder = r"C:\Users\N31937\OneDrive - NGC\Desktop\Coding\DataAnalytics"
    add_custom_matrix_metrics()