# Algorithimic Metric Calculations
import pandas as pd
import numpy as np
import json
from scipy.io import mmread, mminfo
from scipy.linalg import eigh
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import norm, eigsh
from pathlib import Path
import warnings
import gc

# Suppress sparse efficiency warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)

def add_custom_matrix_metrics():
      base_dir = Path('.')
      csv_file_path = base_dir /'matrixdata.csv'
      # Update to specifically look in the 'Matrix Files' subdirectory
      matrix_dir = base_dir / 'Matrix Files'
      try:
            df = pd.read_csv(csv_file_path)
      except FileNotFoundError:
            print(f"Error: Could not find {csv_file_path}")
            return
      df.reset_index(drop=True, inplace=True)
      if 'Name' not in df.columns:
            df.rename(columns={df.columns[2]: 'Name'}, inplace=True)
            df['Name'] = df['Name'].astype(str).str.replace(r'[\m\r]', '', regex=True).str.strip()
      raw_path = [f for f in matrix_dir.rglob('*.mtx') if f.is_file()]
      file_metadata = []
      for path in raw_path:
          try:
              info = mminfo(path)
              nnz = info[2]
              file_metadata.append((path.stem, path, nnz))
          except Exception as e:
              print(f"Skipping {path.stem} for sorting (error: {e}")
      file_metadata.sort(key=lambda x: x[2])
      mtx_files = {stem: path for (stem, path, nnz) in file_metadata}
      #mtx_files = {f.stem: f for f in matrix_dir.rglob('*.mtx') if f.is_file()}
      print(f"Found {len(mtx_files)} .mtx files.\n")
      # Define the exact columns being calculated
      custom_columns = [
            'RCM Bandwidth',
            'Skew-Symmetric Frobenius Norm',
            'Directional Mean Bias',
            'Signed Frobenius Ratio',
            'Gershgorin Discs',
            'Diagonal Dominance Min Ratio',
            'Diagonal Dominance Mean Ratio',
            'Diagonal Dominance Max Ratio',
            'Strictly Diagonally Dominant Row Fraction',
            'Fiedler Value',
            'Brauer Pair Count',
            'Brauer Max Product',
            'Brauer Mean Product',
            'Brauer Min Product',
            'Brauer Max Center Distance',
            'Brauer Mean Center Distance'
      ]
      for col in custom_columns:
            if col not in df.columns:
                  df[col] = np.nan
      for text_col in ['Gershgorin Discs', 'Brauer Unions']:
            if text_col in df.columns:
                  df[text_col] = df[text_col].astype('object')
      modifications = 0
      for name, file_path in mtx_files.items():
            matches = df.index[df['Name'].astype(str).str.strip() == name]
            if len(matches) == 0:
                  continue
            idx = matches[0]
            # Check if there are gaps in any of the custom columns
            needs_calc = any(pd.isna(df.at[idx, col]) for col in custom_columns)
            if not needs_calc:
                  continue
            print(f"Loading and calculating custom metrics for {name}...")
            try:
                  # Load and pre-process
                  A = mmread(file_path).tocsr()
                  n = A.shape[0]
                  calcs = {}
                  # --- Positivity Metrics ---
                  if pd.isna(df.at[idx, 'Signed Frobenius Ratio']):
                        try:
                              data = A.data
                              pos_data = data[data.real > 0]
                              neg_data = data[data.real < 0]
                              # Directional Mean Bias
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
                              # Signed Frobenius Ratio
                              if len(pos_data) > 0 or len(neg_data) > 0:
                                    norm_p = np.sqrt(np.sum(np.abs(pos_data)**2)) if len(pos_data) > 0 else 0.0
                                    norm_n = np.sqrt(np.sum(np.abs(neg_data)**2)) if len(neg_data) > 0 else 0.0
                                    if norm_n > norm_p:
                                          calcs['Signed Frobenius Ratio'] = -(norm_n / norm_p) if norm_p != 0 else -np.inf
                                    else:
                                          calcs['Signed Frobenius Ratio'] = (norm_p / norm_n) if norm_n != 0 else np.inf
                              else:
                                    calcs['Signed Frobenius Ratio'] = 0.0
                        except Exception as e:
                              print(f" -> Error calculating Positivity Metrics: {e}")
                  # --- Fast Diagonal & Row Operations ---
                  if n == A.shape[1]:
                        diag = A.diagonal()
                        abs_A = A.copy()
                        abs_A.data = np.abs(abs_A.data)
                        row_sums = np.array(abs_A.sum(axis=1)).flatten()
                        offdiag_sums = row_sums - np.abs(diag)
                        # --- Skew-Symmetric Frobenius Norm ---
                        if pd.isna(df.at[idx, 'Skew-Symmetric Frobenius Norm']):
                              try:
                                    AS = 0.5 * (A - A.T)
                                    calcs['Skew-Symmetric Frobenius Norm'] = norm(AS, ord='fro')
                              except Exception as e:
                                    print(f" -> Error calculating Skew-Symmetric Frobenius Norm: {e}")
                        # --- Matrix Bandwidth (Reverse Cuthill-McKee) ---
                        if pd.isna(df.at[idx, 'RCM Bandwidth']):
                              try:
                                    perm = reverse_cuthill_mckee(A, symmetric_mode=True)
                                    inv_perm = np.argsort(perm)
                                    rows, cols = A.nonzero()
                                    if len(rows) > 0:
                                          calcs['RCM Bandwidth'] = int(np.max(np.abs(inv_perm[rows] - inv_perm[cols])))
                                    else:
                                          calcs['RCM Bandwidth'] = 0
                              except Exception as e:
                                    print(f" -> Error calculating RCM Bandwidth: {e}")
                        # --- Gershgorin Discs ---
                        try:
                            L = diag - offdiag_sums
                            R = diag + offdiag_sums
                            intervals = np.column_stack((L, R))
                            intervals = intervals[intervals[:, 0].argsort()]
                            merged_gershgorin = []
                            for interval in intervals:
                                l_val = float(interval[0])
                                r_val = float(interval[1])
                                if not merged_gershgorin or merged_gershgorin[-1][1] < l_val:
                                    # They do not overlap -> Distinct union
                                    merged_gershgorin.append([l_val, r_val])
                                else:
                                    # They overlap -> Expand the current union
                                    merged_gershgorin[-1][1] = max(merged_gershgorin[-1][1], r_val)
                            formatted_gersh = []
                            for union in merged_gershgorin:
                                left_bound = round(union[0], 4)
                                right_bound = round(union[1], 4)
                                if left_bound == right_bound:
                                    formatted_gersh.append([left_bound])
                                else:
                                    formatted_gersh.append([left_bound, right_bound])
                            calcs['Gershgorin Discs'] = json.dumps(formatted_gersh, separators=(',', ':'))
                        except Exception as e:
                            print(f" -> Error calculating Gershgorin: {e}")
                            calcs['Gershgorin Discs'] = '[]'
                        # Diagonal Dominance
                        try:
                            ratios = np.divide(np.abs(diag), offdiag_sums, out=np.zeros_like(diag, dtype=float), where=offdiag_sums!=0)
                            ratios[offdiag_sums == 0] = np.inf
                            finite_ratios = ratios[np.isfinite(ratios)]
                            calcs['Diagonal Dominance Min Ratio'] = float(np.min(finite_ratios)) if len(finite_ratios) > 0 else 0.0
                            calcs['Diagonal Dominance Mean Ratio'] = float(np.mean(finite_ratios)) if len(finite_ratios) > 0 else 0.0
                            calcs['Diagonal Dominance Max Ratio'] = float(np.max(finite_ratios)) if len(finite_ratios) > 0 else np.inf
                            calcs['Strictly Diagonally Dominant Row Fraction'] = float(np.sum(ratios > 1) / n) if n > 0 else 0.0
                        except Exception as e:
                            print(f" -> Error calculating Diagonal Dominance: {e}")
                        
                        # --- Brauer Cassini Ovals ---
                        try:
                            if n < 2:
                                calcs.update({k: 0.0 for k in custom_columns if 'Brauer' in k})
                                calcs['Brauer Unions'] = '[]'
                            else:
                                count = 0
                                sum_prod, min_prod, max_prod = 0.0, float('inf'), float('-inf')
                                sum_dist, max_dist = 0.0, float('-inf')
                                chunk_unions = []
                                chunk_size = 500
                                A_csr = A.tocsr()
                                A_csc = A.tocsc()
                                for start_i in range(0, n, chunk_size):
                                    end_i = min(start_i + chunk_size, n)
                                    A_chunk = np.abs(A_csr[start_i:end_i, :].toarray())
                                    A_col_chunk = np.abs(A_csc[:, start_i:end_i].toarray()).T
                                    R_i_j = offdiag_sums[start_i:end_i, None] - A_chunk
                                    R_j_i = offdiag_sums[None, :] - A_col_chunk
                                    prod_chunk = R_i_j * R_j_i
                                    sum_c = diag[start_i:end_i, None] + diag[None, :]
                                    diff_c = diag[start_i:end_i, None] - diag[None, :]
                                    i_idx = np.arange(start_i, end_i)[:, None]
                                    j_idx = np.arange(n)[None, :]
                                    mask = j_idx > i_idx
                                    valid_prods = prod_chunk[mask]
                                    if len(valid_prods) == 0:
                                        continue
                                    valid_sum = sum_c[mask]
                                    valid_diff = diff_c[mask]
                                    count += len(valid_prods)
                                    sum_prod += float(np.sum(valid_prods))
                                    max_prod = max(max_prod, float(np.max(valid_prods)))
                                    min_prod = min(min_prod, float(np.min(valid_prods)))
                                    valid_abs_diff = np.abs(valid_diff)
                                    sum_dist += float(np.sum(valid_abs_diff))
                                    max_dist = max(max_dist, float(np.max(valid_abs_diff)))
                                    D_out = valid_diff**2 + 4 * valid_prods
                                    sqrt_D_out = np.sqrt(D_out)
                                    L_out = (valid_sum - sqrt_D_out) / 2.0
                                    R_out = (valid_sum + sqrt_D_out) / 2.0
                                    D_in = valid_diff**2 - 4 * valid_prods
                                    has_gap = D_in > 0
                                    sqrt_D_in = np.zeros_like(D_in)
                                    sqrt_D_in[has_gap] = np.sqrt(D_in[has_gap])
                                    L_in = (valid_sum - sqrt_D_in) / 2.0
                                    R_in = (valid_sum + sqrt_D_in) / 2.0
                                    intervals = []
                                    for k in range(len(valid_prods)):
                                        if has_gap[k]:
                                            intervals.append([L_out[k].item(), L_in[k].item()])
                                            intervals.append([R_in[k].item(), R_out[k].item()])
                                        else:
                                            intervals.append([L_out[k].item(), R_out[k].item()])
                                    intervals.sort(key=lambda x: x[0])
                                    merged_local = []
                                    for interval in intervals:
                                        l_val, r_val = float(interval[0]), float(interval[1])
                                        if not merged_local or merged_local[-1][1] < l_val:
                                            merged_local.append([l_val, r_val])
                                        else:
                                            merged_local[-1][1] = max(merged_local[-1][1], r_val)
                                    chunk_unions.extend(merged_local)
                                    chunk_unions.sort(key=lambda x: x[0])
                                    final_unions = []
                                    for union in chunk_unions:
                                        if not final_unions or final_unions[-1][1] < union[0]:
                                            final_unions.append(union)
                                        else:
                                            final_unions[-1][1] = max(final_unions[-1][1], union[1])
                                    chunk_unions = final_unions
                                if count > 0:
                                    formatted_brauer = []
                                    for union in chunk_unions:
                                        l_bound = round(union[0], 4)
                                        r_bound = round(union[1], 4)
                                        if l_bound == r_bound:
                                            formatted_brauer.append([l_bound])
                                        else:
                                            formatted_brauer.append([l_bound, r_bound])
                                    calcs['Brauer Unions'] = json.dumps(formatted_brauer, separators=(',', ':'))
                                    calcs['Brauer Pair Count'] = count
                                    calcs['Brauer Max Product'] = float(max_prod)
                                    calcs['Brauer Mean Product'] = float(sum_prod / count)
                                    calcs['Brauer Min Product'] = float(min_prod)
                                    calcs['Brauer Max Center Distance'] = float(max_dist)
                                    calcs['Brauer Mean Center Distance'] = float(sum_dist / count)
                                else:
                                    calcs.update({k: 0.0 for k in custom_columns if 'Brauer' in k})
                                    calcs['Brauer Unions'] = '[]'
                        except Exception as e:
                            print(f" -> Error calculating Brauer Cassini Ovals: {e}")
                        # --- Fiedler Value (Algebraic Connectivity) ---
                        if pd.isna(df.at[idx, 'Fiedler Value']):
                            try:
                                  A_sym = 0.5 * (A + A.T).astype(np.float64)
                                  if n > 2:
                                        try:
                                              # Primary Attempt: Shift-Invert mode (sigma) for rapid convergence near zero
                                              # We use a slight negative offset so the matrix isn't perfectly singular
                                              eigenvalues, _ = eigsh(A_sym, k=2, sigma=-1e-4, maxiter=5000)
                                              # Sort to ensure we grab the 2nd smallest (index 1)
                                              sorted_eigs = np.sort(eigenvalues)
                                              calcs['Fiedler Value'] = float(sorted_eigs[1])
                                        except Exception as arpack_err:
                                              print(f" -> ARPACK failed ({arpack_err}). Attempting dense fallback...")
                                              # Fallback: Brute force exact solver (only safe for matrices < ~5000 rows)
                                              if n <= 5000:
                                                    # eigh computes all eigenvalues, but subset_by_index only grabs the 2 lowest
                                                    dense_eigs = eigh(A_sym.toarray(), eigvals_only=True, subset_by_index=[0, 1])
                                                    calcs['Fiedler Value'] = float(dense_eigs[1])
                                              else:
                                                    print(f" -> Matrix {name} too large for dense fallback. Skipping Fiedler.")
                                                    calcs['Fiedler Value'] = np.nan
                                  else:
                                        calcs['Fiedler Value'] = 0.0
                            except Exception as e:
                                  print(f" -> Error calculating Fiedler Value: {e}")
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
                for v in ['A', 'abs_A_dense', 'R_i_j_matrix', 'R_j_i_matrix', 'prod_matrix', 'discriminant']:
                    if v in locals():
                        del locals()[v]
                gc.collect()
            print(f"\nFinished processing. Added/Updated {modifications} metric values.")

if __name__ == "__main__":
    add_custom_matrix_metrics()

