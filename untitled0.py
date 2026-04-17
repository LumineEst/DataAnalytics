import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import umap
import time
import re
import os
import xgboost as xgb
import shap
import optuna
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import joblib
from joblib import Memory
from pathlib import Path
from scipy.io import mmread
from scipy.sparse.csgraph import reverse_cuthill_mckee, laplacian
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, QuantileTransformer, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold, GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import roc_curve, auc, confusion_matrix, matthews_corrcoef, balanced_accuracy_score, silhouette_score, r2_score, average_precision_score, f1_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.gridspec import GridSpec
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Set up local caching, to prevent redundant computation 
os.makedirs('./cached_computations', exist_ok=True)
cacheDir = Memory(location='./cached_computations', verbose=0)

# Load the raw topological matrix properties and the post-transformation properties.
dfMatrix = pd.read_csv('matrixdata.csv').set_index('Matrix ID')
dfTransforms = pd.read_csv('transforms.csv').set_index('Matrix ID')

# Clean up column names by stripping whitespace
dfTransforms.columns = dfTransforms.columns.str.strip()

# Align the datasets: Ensure we only keep raw matrices that have a corresponding transformed counterpart
dfMatrix = dfMatrix.loc[dfMatrix.index.isin(dfTransforms.index)]
dfMatrix = dfMatrix.loc[:, ~dfMatrix.columns.duplicated()]
dfTransforms = dfTransforms.loc[:, ~dfTransforms.columns.duplicated()]

# Rename verbose columns for cleaner plotting and downstream references
dfMatrix.rename(columns={'Strictly Diagonally Dominant Row Fraction': 'Diagonally Dominant\nRow Fraction'}, inplace=True)
dfTransforms.rename(columns={'Strictly Diagonally Dominant Row Fraction': 'Diagonally Dominant\nRow Fraction'}, inplace=True)

# ==========================
# MATRIX GROUP LOGIC & MASKS
# ==========================
# Reset the index to keep Matrix ID as a standard column for subsetting/tracking.
dfOriginal = dfMatrix.reset_index()

# combine_first is used to backfill missing transformed data with original data.
dfTransformed = dfTransforms.combine_first(dfMatrix).reset_index()

# Boolean Shape flags:
dfOriginal['isSquare'] = dfOriginal['Num Rows'] == dfOriginal['Num Cols']
dfTransformed['isSquare'] = dfTransformed['Num Rows'] == dfTransformed['Num Cols']

# Coerce errors to NaN so that text/garbled data becomes a traceable missing value.
condCol = pd.to_numeric(dfTransformed['Condition Number'], errors='coerce')
msvCol = pd.to_numeric(dfTransformed['Minimum Singular Value'], errors='coerce')

# Defining SVD Failure
dfTransformed['isSvdFailed'] = (condCol.isna() | np.isinf(condCol) | (condCol >= 1e15) | msvCol.isna()).astype(int)

# Dulmage-Mendelsohn (dmperm) blocks dictate structural irreducibility. 
blockCol = pd.to_numeric(dfTransformed['Num Dmperm Blocks'], errors='coerce')
dfTransformed['isIrreducible'] = ((blockCol <= 1) | blockCol.isna()).astype(int)

# Rank Collapse occurs when a matrix does not have full numerical rank, making it singular and uninvertible.
rankCol = pd.to_numeric(dfTransformed['Full Numerical Rank?'], errors='coerce')
dfTransformed['Rank Collapse'] = np.where(rankCol.isna(), np.nan, (rankCol == 0).astype(int))

# Target flag for routing to fast Cholesky factorization.
cholCol = pd.to_numeric(dfTransformed['Cholesky Candidate'], errors='coerce')
dfTransformed['isCholesky'] = (cholCol == 1).astype(int)

binaryTargets = ['isSvdFailed', 'Rank Collapse', 'Positive Definite', 'isCholesky', 'isIrreducible']

brauerCols = ['Brauer Mean Product', 'Brauer Min Product', 'Brauer Max Product', 'Brauer Mean Product (Top)']

# Computation of composite metrics
for df in [dfOriginal, dfTransformed]:
    # Density: Fraction of non-zero elements. Essential baseline for sparsity.
    if set(['Nonzeros', 'Num Rows', 'Num Cols']).issubset(df.columns):
        df['Density'] = df['Nonzeros'].astype(float) / (df['Num Rows'].astype(float) * df['Num Cols'].astype(float))
    
    # RCM Compression: How well did Reverse Cuthill-Mckee reduce the bandwidth?
    if set(['RCM Bandwidth', 'Num Rows']).issubset(df.columns):
        df['RCM Compression Ratio'] = df['RCM Bandwidth'] / df['Num Rows']
    
    # Topological Entropy: A custom metric derived during exploratory analysis
    if set(['RCM Compression Ratio', 'Density']).issubset(df.columns):    
        df['Topological Entropy'] = df['RCM Compression Ratio'] / (df['Density'] + 1e-10)
    
    # Degeneracy: Combines center distances with bias to flag structural anomalies.
    if set(['Brauer Max Center Distance', 'Directional Mean Bias']).issubset(df.columns):
        df['Degeneracy Multiplier'] = df['Brauer Max Center Distance'] * df['Directional Mean Bias']
    
    # Brauer Ratio: Bounds extreme variance. Clipped to 1e35 to prevent float overflow.
    if set(['Brauer Min Product', 'Brauer Max Product']).issubset(df.columns):
        df['Brauer Ratio'] = (np.sqrt(df['Brauer Max Product']) / (np.sqrt(df['Brauer Min Product'] + 1e-10))).replace([np.inf, -np.inf], 1e35)
        df['Brauer Ratio'] = np.clip(df['Brauer Ratio'], a_min=1e-35, a_max=1e35)   
    
    # Square Root Normalization: Tames extreme outliers in Brauer metrics.
    for baseCol in brauerCols:
        if baseCol in df.columns:
            safeNumericCol = pd.to_numeric(df[baseCol], errors='coerce').fillna(0)
            df[f'{baseCol} Sqrt'] = np.sqrt(np.maximum(safeNumericCol, 0))

def applyPositivityMask(df):  
    # Flags matrices that have blown out to infinity on specific directional/positivity metrics.
    mask = pd.Series(False, index=df.index)
    for col in ['Directional Mean Bias', 'Signed Frobenius Ratio']:
        if col in df.columns:
            mask |= np.isinf(pd.to_numeric(df[col], errors='coerce'))
    df['isInfinitePositivity'] = mask
    return df

dfOriginal = applyPositivityMask(dfOriginal)
dfTransformed = applyPositivityMask(dfTransformed)

# Define real-world scientific domains based on group descriptions on Matrix Market
optimizationGroups = ['LPnetlib', 'GHS_indef', 'Schenk_IBMSDS', 'Schenk_IBMNA', 'Rajat']
physicsGroups = ['HB', 'Freescale']
networkGroups = ['LAW', 'SNAP', 'FlowIPM22', 'Pajek']

def categorizeGroup(g):
    if g in optimizationGroups: return 'Optimization'
    if g in physicsGroups: return 'Applied Physics'
    if g in networkGroups: return 'Network Graphs'
    return 'Other'

dfOriginal['matrixGroup'] = dfOriginal['Group'].apply(categorizeGroup)
dfTransformed['matrixGroup'] = dfTransformed['Group'].apply(categorizeGroup)

# Handle structural rank degeneracy (Difference between physical rows and mathematically independent rows)
if set(['Structural Rank', 'Rank']).issubset(dfOriginal.columns):
    dfOriginal['Rank Degeneracy'] = dfOriginal['Structural Rank'] - dfOriginal['Rank']
if set(['Structural Rank', 'Rank']).issubset(dfTransformed.columns):
    dfTransformed['Rank Degeneracy'] = dfTransformed['Structural Rank'] - dfTransformed['Rank'] 

# Create isolated subsets for geometry-specific models
dfSquareOriginal = dfOriginal[dfOriginal['isSquare'] == True].copy()
dfSquareTransformed = dfTransformed[dfTransformed['isSquare'] == True].copy()
dfRectangularTransformed = dfTransformed[dfTransformed['isSquare'] == False].copy()

# Double check that rank collapse is populated across all subsets
for df in [dfSquareOriginal, dfSquareTransformed, dfRectangularTransformed, dfTransformed, dfOriginal]:
    if 'Full Numerical Rank?' in df.columns:
        rankColRF = pd.to_numeric(df['Full Numerical Rank?'], errors='coerce')
        df['Rank Collapse'] = np.where(rankColRF.isna(), np.nan, (rankColRF == 0).astype(int))

# ====================================================
# WILCOXON SIGNED-RANK TEST OF PRE-POST TRANSFORM DATA
# ====================================================
# A non-parametric equivalent of a paired t-test, as topological data is highly skewed (non-normal) 
# and because we are comparing the *exact same* matrix before and after transformation.

keys = ['Matrix ID', 'Name', 'Group', 'Group.1']
overlapCols = [c for c in dfMatrix.columns if c in dfTransforms.columns and c not in keys]
wilcoxonResults = []

if len(overlapCols) > 0:
    for col in overlapCols:
        # Align data strictly by dropping rows where the computation failed in either state
        pairedDf = pd.DataFrame({
            'Original': pd.to_numeric(dfSquareOriginal[col], errors='coerce'),
            'Transformed': pd.to_numeric(dfSquareTransformed[col], errors='coerce')
        }).replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(pairedDf) > 0:
            stat, p = stats.wilcoxon(pairedDf['Original'], pairedDf['Transformed'])
            wilcoxonResults.append({'Metric': col, 'WStat': stat, 'PValue': p})
            
    dfWilcoxon = pd.DataFrame(wilcoxonResults).sort_values(by='PValue')

plt.figure(figsize=(10, max(4, len(dfWilcoxon) * 0.5)))
# Visual scaling: Convert p-values to -log10. 
dfWilcoxon['LogP'] = -np.log10(dfWilcoxon['PValue'] + 1e-300)

sns.barplot(data=dfWilcoxon, x='LogP', y='Metric', palette='viridis')
plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05 Threshold')
plt.title('Wilcoxon Signed-Rank Test Significance (-log10 P-Value)\n(Impact of Transformations)')
plt.xlabel('-log10(P-Value)')
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# KRUSKAL-WALLIS H-TEST OF MATRIX GROUPINGS
# =========================================
# A non-parametric equivalent of an ANOVA, to test if 3 or more *independent* groups
# originate from the same statistical distribution.

hTestResults = []
configs = {'Square Transformed': dfSquareTransformed, 'Rectangular Transformed': dfRectangularTransformed}
masks = ['isInfinitePositivity', 'matrixGroup']
ignoreCols = keys + ['Kind', 'Type', 'Author', 'isSquare', 'isInfinitePositivity', 'matrixGroup', 'Gershgorin Discs']

for configName, df in configs.items():
    # Filter for purely numerical columns to run the stats test on
    numCols = [c for c in df.columns if c not in ignoreCols and pd.api.types.is_numeric_dtype(df[c])]
    
    for mask in masks:
        uniqueGroups = df[mask].dropna().unique()
        if len(uniqueGroups) < 2: continue # Needs at least 2 groups to compare
        
        for metric in numCols:
            groupsData = []
            # Gather valid numeric data for each categorical group
            for g in uniqueGroups:
                data = df[df[mask] == g][metric].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) > 0: groupsData.append(data)
                
            # Execute the test if we successfully gathered data for multiple groups
            if len(groupsData) >= 2:
                try:
                    stat, p = stats.kruskal(*groupsData)
                    hTestResults.append({'DataFrame': configName, 'SplitBy': mask, 'Metric': metric, 'HStat': stat, 'PValue': p})
                except Exception: pass
                
dfHTest = pd.DataFrame(hTestResults).sort_values(by='PValue')

# Visualize H-Test Significance via Dual Heatmap
dfHTest['LogP'] = -np.log10(dfHTest['PValue'] + 1e-300)
labelMap = {'isInfinitePositivity': 'Positivity\nSkew', 'matrixGroup': 'Matrix\nGroup'}
dfHTest['SplitByShort'] = dfHTest['SplitBy'].map(labelMap)

dfSquare = dfHTest[dfHTest['DataFrame'] == 'Square Transformed']
dfRect = dfHTest[dfHTest['DataFrame'] == 'Rectangular Transformed']

fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(dfHTest['Metric'].unique()) * 0.4)))

# Subplot 1: Square Matrices
pivotSquareLogP = dfSquare.pivot_table(index='Metric', columns='SplitByShort', values='LogP', fill_value=0)
pivotSquarePVal = dfSquare.pivot_table(index='Metric', columns='SplitByShort', values='PValue', fill_value=1.0)
# The color intensity maps to the -log10 value (so hotter = more significant)
sns.heatmap(pivotSquareLogP, annot=pivotSquarePVal, fmt=".1e", cmap='magma', ax=axes[0], cbar_kws={'label': '-log10(P-Value)'})
axes[0].set_title('Square Matrices', fontsize=14, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Metric', fontsize=12)

# Subplot 2: Rectangular Matrices
pivotRectLogP = dfRect.pivot_table(index='Metric', columns='SplitByShort', values='LogP', fill_value=0)
pivotRectPVal = dfRect.pivot_table(index='Metric', columns='SplitByShort', values='PValue', fill_value=1.0)
sns.heatmap(pivotRectLogP, annot=pivotRectPVal, fmt=".1e", cmap='magma', ax=axes[1], cbar_kws={'label': '-log10(P-Value)'})
axes[1].set_title('Rectangular Matrices', fontsize=14, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('')

plt.suptitle('Kruskal-Wallis H-Test Significance by Core Matrix Groupings\n(Identifying Domain-Specific Topologies)', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# =======================
# HIERARCHICAL CLUSTERMAP
# =======================

# Standardize the 'Rank Collapse' flag across all datasets if the raw column exists.
for df in [dfSquareOriginal, dfSquareTransformed, dfRectangularTransformed]:
    if 'Full Numerical Rank?' in df.columns:
        df['Rank Collapse'] = (df['Full Numerical Rank?'] == 0).astype(int)

# Define the primary algorithmic failure modes and solvability constraints
solvabilityTargets = [
    'Condition Number', 'Matrix Norm', 'Num Dmperm Blocks', 'Strongly Connect Components',
    'Rank Collapse', 'Positive Definite', 'Cholesky Candidate'
]

configs = [('Rectangular Transformed', dfRectangularTransformed), ('Square Original', dfSquareOriginal), ('Square Transformed', dfSquareTransformed)]
numTargets = len(solvabilityTargets)
numConfigs = len(configs)

# Dynamically extract strictly numeric features that aren't target variables or metadata keys
transformCols = [c for c in dfTransforms.columns if pd.api.types.is_numeric_dtype(dfTransforms[c]) and c not in keys + solvabilityTargets + ['isSquare']]

# Master list of predictors: Combine raw numeric transforms with our custom engineered topological heuristics
masterPredictors = transformCols + ['Density', 'RCM Compression Ratio', 'Topological Entropy', 'Degeneracy Multiplier', 'Brauer Ratio']

clusterCols = [c for c in masterPredictors if c in dfTransformed.columns]

# Aggregation Strategy: Group matrices into "Archetypes" based on Domain Group, Positivity Skew, and Shape.
groupedDf = dfTransformed.groupby(['matrixGroup', 'isInfinitePositivity', 'isSquare'])[clusterCols].median()

# Drop features that are entirely null across these archetypes
groupedDf = groupedDf.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')

# Create readable, multi-line labels for the Clustermap Y-axis
groupedDf.index = [f"{g[0]}\nPosInf: {g[1]}\nSq: {g[2]}" for g in groupedDf.index]

# Data Prep for Distance Metrics: Ward Linkage requires clean, complete numerical data without infinities.
imputer = SimpleImputer(strategy='median')
imputedData = imputer.fit_transform(groupedDf)
imputedData = np.clip(imputedData, -1e35, 1e35)

# Standardize features so a variable measured in millions doesn't overpower a ratio measured between [0,1].
scaler = StandardScaler()
scaledData = scaler.fit_transform(imputedData)
scaledDf = pd.DataFrame(scaledData, index=groupedDf.index, columns=groupedDf.columns)

# Drop Zero/Low-Variance columns (which break distance calculations) and clip extreme Z-scores for readability
scaledDf = scaledDf.loc[:, scaledDf.var() > 0.01].clip(lower=-5, upper=5)

plt.figure(figsize=(14, 10))
# The Clustermap uses Ward linkage to minimize the variance within clusters, grouping archetypes that share similar topological metric signatures.
cg = sns.clustermap(scaledDf.T, cmap='coolwarm', metric='euclidean', method='ward', figsize=(16, 12), cbar_kws={'label': 'Z-Score'}, xticklabels=True, yticklabels=True)
cg.fig.suptitle("Hierarchical Clustermap of Metrics (Aggregated by 3 Masks)\n(Identifying Covariant Feature Blocks)", fontsize=16, fontweight='bold', x=0.98, y=0.98, ha='right')
plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.show()

# ==========================================
# RANDOM FOREST & SUBPLOTTED PARALLEL COORDS
# ==========================================

xgbFeatures = list (masterPredictors)
for df in [dfTransformed, dfOriginal, dfSquareOriginal, dfSquareTransformed, dfRectangularTransformed]:
    for col in masterPredictors:
        if col in df.columns:
            infCol = f"Inf {col}"
            nanCol = f"Missing {col}"
            df[infCol] = np.isinf(pd.to_numeric(df[col], errors='coerce')).astype(int)
            df[nanCol] = df[col].isna().astype(int)
            
for col in masterPredictors:
    if col in dfTransformed.columns:
        xgbFeatures.extend([f"Inf {col}", f"Missing {col}"])

fig, axes = plt.subplots(nrows=numTargets * 2, ncols=numConfigs, figsize=(26, 80))

for tIdx, targetCol in enumerate(solvabilityTargets):
    for cIdx, (configName, dfConfig) in enumerate(configs):
        axPc = axes[tIdx * 2, cIdx]
        axBar = axes[tIdx * 2 + 1, cIdx]
        featureCols = [c for c in masterPredictors if c in dfConfig.columns]
        
        if 'isInfinitePositivity' in dfConfig.columns and 'isInfinitePositivity' not in featureCols: featureCols.append('isInfinitePositivity')
        
        pcaFeatures = ['Degeneracy Multiplier', 'Topological Entropy', 'Directional Mean Bias', 'Signed Frobenius Ratio', 'Brauer Mean Center Distance']
        tsneFeatures = ['Degeneracy Multiplier', 'Signed Frobenius Ratio', 'Topological Entropy', 'Brauer Mean Center Distance', 'Brauer Mean Product (Top)']
        umapFeatures = ['Topological Entropy', 'Brauer Mean Product (Top)', 'Signed Frobenius Ratio', 'Directional Mean Bias', 'Brauer Ratio']
        
        # Non-square matrices physically cannot be evaluated for "Positive Definite" or "Cholesky" conditions.
        # So Create alternative Charts which evaluate significant combinations to avoid wasting chart space.
        if configName == 'Rectangular Transformed' and targetCol in ['Positive Definite', 'Cholesky Candidate']:
            targetSequence = ['Rank Collapse', 'Matrix Norm', 'Condition Number', 'Num Dmperm Blocks', 'Strongly Connect Components', 'Cholesky Candidate']
            validTargetSeq = [c for c in targetSequence if c in dfConfig.columns]
            
            if targetCol == 'Positive Definite':
                customPlots = [(axPc, validTargetSeq, "Target Sequence"), (axBar, pcaFeatures, "PCA Top Loadings")]
            else:
                customPlots = [(axPc, tsneFeatures, "t-SNE Top Loadings"), (axBar, umapFeatures, "UMAP Top Loadings")]
                
            for currAx, currCols, currTitle in customPlots:
                validCols = [c for c in currCols if c in dfConfig.columns]
                if len(validCols) >= 2:
                    tempDf = dfConfig[validCols + ['matrixGroup']].copy().replace([np.inf, -np.inf], [1e35, -1e35]).dropna()
                    if len(tempDf) > 4:
                        # Use QuantileTransformer to map skewed data evenly across [0,1], maximizing visual clarity in the Parallel Coordinates plot.
                        dataScaler = QuantileTransformer(output_distribution='uniform', random_state=42)
                        yLabelText = "Uniform Quantiles"
                        for c in validCols:
                            tempDf[c] = dataScaler.fit_transform(tempDf[[c]])
                            
                        pd.plotting.parallel_coordinates(tempDf, 'matrixGroup', cols=validCols, colormap='plasma', alpha=0.6, ax=currAx)
                        currAx.set_title(f"Rectangular: {currTitle}", fontsize=14, fontweight='bold')
                        currAx.set_xticklabels([c.replace(' ', '\n') for c in validCols], rotation=0, ha='center', fontsize=10)
                        currAx.set_ylabel(yLabelText, fontsize=11)
                        currAx.legend(loc='upper right')
                    else:
                        currAx.set_axis_off()
                else:
                    currAx.set_axis_off()
            continue # Skip the Random Forest generation for these specific intercept cells

        # ==================== DATA PREP ====================
        rfData = dfConfig[[targetCol, 'matrixGroup'] + featureCols].copy()
        if 'isInfinitePositivity' in rfData.columns: rfData['isInfinitePositivity'] = rfData['isInfinitePositivity'].astype(int)
        
        if targetCol not in ['matrixGroup', 'isInfinitePositivity']:
            rfData[targetCol] = pd.to_numeric(rfData[targetCol], errors='coerce').astype(float)
            
        rfData = rfData.replace([np.inf, -np.inf], [1e35, -1e35])
        
        for col in featureCols:
            if col != 'isInfinitePositivity':
                safeCol = pd.to_numeric(rfData[col], errors='coerce').astype(float)
                # Apply symmetric log1p to compress extreme variance without destroying negative values
                rfData[col] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
        
        rfData = rfData.dropna(subset=[targetCol, 'matrixGroup'])
        
        # ==================== MODEL TRAINING ====================
        X = rfData[featureCols].values
        # Dynamically route to Classification vs Regression based on the mathematical nature of the target
        is_classification = targetCol in ['Rank Collapse', 'Positive Definite', 'Cholesky Candidate']
        
        if is_classification:
            y = rfData[targetCol].astype(int).values
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42)
            cvFolds = KFold(n_splits=20, shuffle=True, random_state=42) # 20 Folds = Strict evaluation
            cvScores = cross_val_score(rf, X, y, cv=cvFolds, scoring='accuracy')
            metricLabel = "Accuracy"
        else:
            # Compress continuous targets that scale exponentially (like Condition Number) so the MSE loss function isn't dominated by massive outliers.
            if targetCol in ['Condition Number', 'Matrix Norm', 'Num Dmperm Blocks', 'Strongly Connect Components']:
                y = np.log1p(np.clip(rfData[targetCol].values, 0, 1e35))
            else:
                y = rfData[targetCol].values
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42)
            cvFolds = KFold(n_splits=20, shuffle=True, random_state=42)
            cvScores = cross_val_score(rf, X, y, cv=cvFolds, scoring='r2')
            metricLabel = "R²"
            
        accMean = np.mean(cvScores)
        rf.fit(X, y)
        
        # Extract the Top 5 most predictive drivers for this specific target
        importances = pd.Series(rf.feature_importances_, index=featureCols).sort_values(ascending=False)
        topFeatures = importances.head(5).index.tolist()
        topImportances = importances.head(5).values

        # ==================== PLOTTING ====================
        pcData = rfData[topFeatures + [targetCol, 'matrixGroup']].copy()
        
        for col in topFeatures + [targetCol]:
            if col != 'isInfinitePositivity': pass
            # MinMax Scale everything to [0,1] so vastly different metrics share the same Parallel Coords Y-Axis
            pcData[col] = minmax_scale(pcData[col])
            
        renameMap = {col: col.replace(' ', '\n') for col in topFeatures + [targetCol]}
        pcData = pcData.rename(columns=renameMap)
        newTopFeatures = [renameMap[col] for col in topFeatures]
        newTargetCol = renameMap[targetCol]
        
        # Sorting Strategy: Count the size of each scientific group and plot the largest groups first to mitigate obfuscation.
        groupCounts = pcData['matrixGroup'].value_counts()
        pcData['count'] = pcData['matrixGroup'].map(groupCounts)
        pcData = pcData.sort_values('count', ascending=False).drop(columns=['count'])
        ordered_cols = [newTargetCol] + newTopFeatures
        
        pd.plotting.parallel_coordinates(pcData, 'matrixGroup', cols=ordered_cols, colormap='viridis', alpha=0.6, ax=axPc)

        if tIdx == 0:
            axPc.set_title(f"{configName}\n\nTarget: {targetCol}", fontsize=18, fontweight='bold')
        else:
            axPc.set_title(f"Target: {targetCol}", fontsize=16, fontweight='bold')
            
        axPc.set_xticklabels(axPc.get_xticklabels(), rotation=0, ha='center', fontsize=12)
        axPc.set_ylabel("Normalized Value [0, 1]", fontsize=12)
        axPc.legend(loc='upper right')
        
        # Feature Importance Bar Plot
        barLabels = [f.replace(' ', '\n') for f in topFeatures]
        sns.barplot(x=topImportances, y=barLabels, ax=axBar, palette='mako')
        
        accColor = 'darkred' if accMean < 0.5 else 'darkgreen'
        axBar.set_title(f"Model {metricLabel} (5-Fold CV): {accMean:.2f}", fontsize=14, fontweight='bold', color=accColor)
        axBar.set_xlabel("Relative Importance Contribution", fontsize=12)
        axBar.set_xlim(0, 1.0)
        
        if tIdx < numTargets - 1:
            # Draw a thick visual divider line between targets in the massive subplot grid
            axBar.axhline(y=len(barLabels), color='black', linewidth=4, alpha=0.5, clip_on=False)

plt.tight_layout()
plt.show()

# ========================
# DIMENSIONALITY REDUCTION
# ========================

def getMultiScaleTrust (xData, embData):
    validNeighbors = [n for n in [5, 10, 15, 20, 25, 30] if n <= len(xData) - 1]
    if not validNeighbors: return 0.0
    return np.mean([trustworthiness(xData, embData, n_neighbors=n) for n in validNeighbors])
    
def prepareManifoldData(df, cols):
    tempData = df[cols].copy()
    tempData = tempData.replace([np.inf, -np.inf], [1e35, -1e35])
    if 'Minimum Singular Value' in tempData.columns: tempData['Minimum Singular Value'] = tempData['Minimum Singular Value'].fillna(0.0)
    if 'Condition Number' in tempData.columns: tempData['Condition Number'] = tempData['Condition Number'].fillna(1e35)
    for colName in cols:
        safeCol = pd.to_numeric(tempData[colName], errors='coerce').astype(float)
        tempData[colName] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
    imputedData = SimpleImputer(strategy='median', add_indicator=True).fit_transform(tempData)
    return MinMaxScaler().fit_transform(imputedData)

def getEmbeddingR2(xData, embData):
    if embData.shape[1] < 2: return 0.0
    lrModel = LinearRegression().fit(embData, xData)
    return lrModel.score(embData, xData)

def getAdjR2(r2Score, nSamples, pFeatures):
    if nSamples - pFeatures - 1 <= 0: return 0.0
    return 1 - (1 - r2Score) * (nSamples - 1) / (nSamples - pFeatures - 1)

def getManifoldImportance(xData, embId, cols):
    safeX = np.array(xData).astype(float)
    miScore = mutual_info_regression(safeX, embId, random_state=42)
    if miScore.max() > 0:
        miScore = miScore / miScore.max()
    return pd.Series(miScore[:len(cols)], index=cols)

def dropCovariantFeatures(df, cols, labels, threshold=0.90):
    # 1. Calculate baseline MI for all features against the target domains
    safeData = df[cols].replace([np.inf, -np.inf], np.nan)
    impData = SimpleImputer(strategy='median', add_indicator=True).fit_transform(safeData)
    miScores = mutual_info_classif(impData, labels, random_state=42)
    miDict = dict(zip(cols, miScores))
    # 2. Calculate Spearman correlation matrix
    corrMatrix = safeData.corr(method='spearman').abs()
    droppedCols = set()
    # 3. Iterate through the upper triangle of the correlation matrix
    for i in range(len(corrMatrix.columns)):
        for j in range(i + 1, len(corrMatrix.columns)):
            colA = corrMatrix.columns[i]
            colB = corrMatrix.columns[j]
            if colA in droppedCols or colB in droppedCols:
                continue
            if corrMatrix.iloc[i, j] > threshold:
                # Tie-breaker: Drop the one with less Mutual Information to the target
                if miDict[colA] >= miDict[colB]:
                    droppedCols.add(colB)
                else:
                    droppedCols.add(colA)
    survivingCols = [c for c in cols if c not in droppedCols]
    print(f"[Pre-Filter] Dropped {len(droppedCols)} highly covariant features: {list(droppedCols)}")
    return survivingCols

targetGroups = ['Optimization', 'Applied Physics', 'Network Graphs']
dfDim = dfTransformed[dfTransformed['matrixGroup'].isin(targetGroups)].copy()
baseCols = [c for c in masterPredictors if c in dfDim.columns]
labels = dfDim['matrixGroup'].values
umapLabels = pd.factorize(labels)[0]

dimCheckpointPath = './cached_computations/dim_reduction_checkpoint.joblib'

if os.path.exists(dimCheckpointPath):
    print("CONSOLE LOG: Loading pre-computed dimensionality reduction manifolds...")
    dimCheckpoint = joblib.load(dimCheckpointPath)
    dfDim = dimCheckpoint['dfDim']
    bestResults = dimCheckpoint['bestResults']
    pcaLoadingsDf, tsneLoadingsDf, umapLoadingsDf = dimCheckpoint['pcaLoadingsDf'], dimCheckpoint['tsneLoadingsDf'], dimCheckpoint['umapLoadingsDf']
    pcaUnsupCols, tsneUnsupCols, umapUnsupCols = dimCheckpoint['pcaUnsupCols'], dimCheckpoint['tsneUnsupCols'], dimCheckpoint['umapUnsupCols']
    pcaUnsupScore, tsneUnsupScore, umapUnsupScore = dimCheckpoint['pcaUnsupScore'], dimCheckpoint['tsneUnsupScore'], dimCheckpoint['umapUnsupScore']
    pcaUnsupLoadingsDf, tsneUnsupLoadingsDf, umapUnsupLoadingsDf = dimCheckpoint['pcaUnsupLoadingsDf'], dimCheckpoint['tsneUnsupLoadingsDf'], dimCheckpoint['umapUnsupLoadingsDf']
    pcaUnsupR2, tsneUnsupR2, umapUnsupR2 = dimCheckpoint['pcaUnsupR2'], dimCheckpoint['tsneUnsupR2'], dimCheckpoint['umapUnsupR2']
else:
    print("CONSOLE LOG: Executing dimensionality reduction sweeps...")
    bestResults = {'PCA': {'score': -1}, 'TSNE': {'score': -1}, 'UMAP': {'score': -1}}
    baseCols = dropCovariantFeatures(dfDim, baseCols, umapLabels, threshold=0.85)
    
    # ============================
    # TARGETED / SUPERVISED MODELS
    # ============================
    
    def scoreSubset(algo, cols):
        scaledData = prepareManifoldData(dfDim, cols)
        bestSc = -1
        try:
            if algo == 'PCA':
                embData = KernelPCA(n_components=2, kernel='cosine', random_state=42).fit_transform(scaledData)
            elif algo == 'TSNE':
                embData = TSNE(n_components=2, perplexity=min(30, len(scaledData)-1), metric='cosine', init='pca', random_state=42).fit_transform(scaledData)
            elif algo == 'UMAP':
                embData = umap.UMAP(n_components=2, metric='cosine', random_state=42).fit_transform(scaledData, y=umapLabels)
            bestSc = getMultiScaleTrust(scaledData, embData)
        except Exception as e:
            print(f"[Warning] Supervised {algo} failed: {e}")
        return bestSc
    
    
    for algo in ['PCA', 'TSNE', 'UMAP']:
        currentCols = list(baseCols)
        bestScore = scoreSubset(algo, currentCols)
        improved = True
        while improved and len(currentCols) > 2:
            improved = False
            bestStepScore = -1
            colToDrop = None
            for col in currentCols:
                testCols = [c for c in currentCols if c != col]
                sc = scoreSubset(algo, testCols)
                if sc > bestStepScore:
                    bestStepScore = sc
                    colToDrop = col
            if bestStepScore > bestScore:
                bestScore = bestStepScore
                currentCols.remove(colToDrop)
                improved = True
        droppedCols = [c for c in baseCols if c not in currentCols]
        improved = True
        while improved and len(droppedCols) > 0:
            improved = False
            bestStepScore = -1
            colToAdd = None
            for col in droppedCols:
                testCols = currentCols + [col]
                sc = scoreSubset(algo, testCols)
                if sc > bestStepScore:
                    bestStepScore = sc
                    colToAdd = col
            if bestStepScore > bestScore:
                bestScore = bestStepScore
                currentCols.append(colToAdd)
                droppedCols.remove(colToAdd)
                improved = True
    
        bestResults[algo] = {'prunedCols': currentCols, 'score': bestScore}
        print(f"[{algo}] Pruned to {len(currentCols)} vars | Score: {bestScore:.3f}")
    
    # PCA Execution
    pcaCols = bestResults['PCA']['prunedCols']
    pcaScaled = prepareManifoldData(dfDim, pcaCols)
    pcaEmb = KernelPCA(n_components=2, kernel='cosine', random_state=42).fit_transform(pcaScaled)
    pcaTargetR2 = getEmbeddingR2(pcaScaled, pcaEmb)
    bestResults['PCA'].update({'emb': pcaEmb, 'scaled': pcaScaled, 'r2': pcaTargetR2})
    
    # TSNE Execution
    tsneCols = bestResults['TSNE']['prunedCols']
    tsneScaled = prepareManifoldData(dfDim, tsneCols)
    baseTsneEmb = TSNE(n_components=2, perplexity=min(30, len(tsneScaled)-1), random_state=42).fit_transform(tsneScaled)
    bestResults['TSNE'].update({'emb': baseTsneEmb, 'scaled': tsneScaled})
    maxPerp = min(50, len(dfDim) - 1)
    for perp in [10, 30, min(50, maxPerp)]:
        try:
            tempEmb = TSNE(n_components=2, perplexity=perp, metric='cosine', random_state=42).fit_transform(tsneScaled)
            sc = getMultiScaleTrust(tsneScaled, tempEmb)
            if sc > bestResults['TSNE'].get('score', -1):
                bestResults['TSNE'].update({'emb': tempEmb, 'scaled': tsneScaled, 'score': sc, 'perp': perp})
        except: pass
    tsneTargetR2 = getEmbeddingR2(tsneScaled, bestResults['TSNE']['emb'])
    bestResults['TSNE'].update({'r2': tsneTargetR2})
    
    # UMAP Execution
    umapCols = bestResults['UMAP']['prunedCols']
    umapScaled = prepareManifoldData(dfDim, umapCols)
    umapBaseEmb = umap.UMAP(n_components=2, metric='cosine', random_state=42).fit_transform(umapScaled, y=umapLabels)
    bestResults['UMAP'].update({'emb': umapBaseEmb, 'scaled': umapScaled})
    for nNeighbors in [5, 15, 30]:
        for minDist in [0.01, 0.1, 0.5]:
            try:
                tempEmb = umap.UMAP(n_components=2, n_neighbors=nNeighbors, metric='cosine', min_dist=minDist, random_state=42).fit_transform(umapScaled, y=umapLabels)
                sc = getMultiScaleTrust(umapScaled, tempEmb)
                if sc > bestResults['UMAP'].get('score', -1):
                    bestResults['UMAP'].update({'emb': tempEmb, 'scaled': umapScaled, 'score': sc, 'nNeighbors': nNeighbors, 'minDist': minDist})
            except: pass
    umapTargetR2 = getEmbeddingR2(umapScaled, bestResults['UMAP']['emb'])
    bestResults['UMAP'].update({'r2': umapTargetR2})
    
    # Extract Loadings
    pcaModelSupervised = PCA(n_components=2, random_state=42).fit(bestResults['PCA']['scaled'])
    
    pcaLoadingsDf = pd.DataFrame({
        'Comp 1': getManifoldImportance(bestResults['PCA']['scaled'], bestResults['PCA']['emb'][:, 0], bestResults['PCA']['prunedCols']),
        'Comp 2': getManifoldImportance(bestResults['PCA']['scaled'], bestResults['PCA']['emb'][:, 1], bestResults['PCA']['prunedCols'])
    }).sort_values(by='Comp 1', ascending=False).head(8)
    
    tsneLoadingsDf = pd.DataFrame({
        'Comp 1': getManifoldImportance(bestResults['TSNE']['scaled'], bestResults['TSNE']['emb'][:, 0], bestResults['TSNE']['prunedCols']),
        'Comp 2': getManifoldImportance(bestResults['TSNE']['scaled'], bestResults['TSNE']['emb'][:, 1], bestResults['TSNE']['prunedCols'])
    }).sort_values(by='Comp 1', ascending=False).head(8)
    
    umapLoadingsDf = pd.DataFrame({
        'Comp 1': getManifoldImportance(bestResults['UMAP']['scaled'], bestResults['UMAP']['emb'][:, 0], bestResults['UMAP']['prunedCols']),
        'Comp 2': getManifoldImportance(bestResults['UMAP']['scaled'], bestResults['UMAP']['emb'][:, 1], bestResults['UMAP']['prunedCols'])
    }).sort_values(by='Comp 1', ascending=False).head(8)
    
    # ============================
    # UNSUPERVISED BASELINE MODELS
    # ============================
    def getUnsupScore(cols):
        scaledData = prepareManifoldData(dfDim, cols)
        try:
            pcaModelUnsup = KernelPCA(n_components=2, kernel='cosine', random_state=42)
            embData = pcaModelUnsup.fit_transform(scaledData)
            trustScore = getMultiScaleTrust(scaledData, embData)
            # Use Mutual Information for non-linear loadings
            loadings = getManifoldImportance(scaledData, embData[:, 0], cols) + getManifoldImportance(scaledData, embData[:, 1], cols)
            return trustScore, loadings, scaledData
        except Exception as e:
            print(f"[Warning] Unsupervised PCA failed: {e}")
    
    uCols = list(baseCols)
    bestUVar, bestULoadings, uScaled = getUnsupScore(uCols)
    
    improved = True
    while improved and len(uCols) > 4:
        improved = False
        leastImportantIdx = np.argmin(bestULoadings)
        colToDrop = uCols[leastImportantIdx]
        testCols = [c for c in uCols if c != colToDrop]
        varSc, loadings, scaledData = getUnsupScore(testCols)
        if varSc > bestUVar:
            bestUVar = varSc
            bestULoadings = loadings
            uScaled = scaledData
            uCols.remove(colToDrop)
            improved = True
    
    uDropped = [c for c in baseCols if c not in uCols]
    improved = True
    while improved and len(uDropped) > 0:
        improved = False
        bestStepVar = -1
        colToAdd = None
        stepLoadings = None
        stepScaled = None
        for col in uDropped:
            testCols = uCols + [col]
            varSc, loadings, scaledData = getUnsupScore(testCols)
            if varSc > bestStepVar:
                bestStepVar = varSc
                colToAdd = col
                stepLoadings = loadings
                stepScaled = scaledData
        if bestStepVar > bestUVar:
            bestUVar = bestStepVar
            bestULoadings = stepLoadings
            uScaled = stepScaled
            uCols.append(colToAdd)
            uDropped.remove(colToAdd)
            improved = True
    
    print(f"[Unsupervised] Pruned to {len(uCols)} vars | 2D Explained Variance: {bestUVar:.3f}")
    
    pcaUnsupModel = KernelPCA(n_components=2, kernel='cosine', random_state=42).fit(uScaled)
    pcaUnsupEmb = pcaUnsupModel.transform(uScaled)
    tsneUnsupEmb = TSNE(n_components=2, perplexity=min(30, len(uScaled)-1), metric='cosine', init='pca', random_state=42).fit_transform(uScaled)
    umapUnsupEmb = umap.UMAP(n_components=2, metric='cosine', random_state=42).fit_transform(uScaled)
    
    pcaUnsupScore = getMultiScaleTrust(uScaled, pcaUnsupEmb)
    tsneUnsupScore = getMultiScaleTrust(uScaled, tsneUnsupEmb)
    umapUnsupScore = getMultiScaleTrust(uScaled, umapUnsupEmb)
    
    pcaUnsupLoadingsDf = pd.DataFrame({
        'Comp 1': getManifoldImportance(uScaled, pcaUnsupEmb[:, 0], uCols),
        'Comp 2': getManifoldImportance(uScaled, pcaUnsupEmb[:, 1], uCols)
    }).sort_values(by='Comp 1', ascending=False).head(8)
    
    tsneUnsupLoadingsDf = pd.DataFrame({
        'Comp 1': getManifoldImportance(uScaled, tsneUnsupEmb[:, 0], uCols),
        'Comp 2': getManifoldImportance(uScaled, tsneUnsupEmb[:, 1], uCols)
    })
    
    umapUnsupLoadingsDf = pd.DataFrame({
        'Comp 1': getManifoldImportance(uScaled, umapUnsupEmb[:, 0], uCols),
        'Comp 2': getManifoldImportance(uScaled, umapUnsupEmb[:, 1], uCols)
    })
    
    miThreshold = 0.01
    
    def applyMiPruning(loadingsDf, baseColsList, algoName, baseEmb, baseScore):
        aliveFeatures = loadingsDf[(loadingsDf['Comp 1'] > miThreshold) | (loadingsDf['Comp 2'] > miThreshold)].index.tolist()
        prunedCols = [c for c in baseColsList if c in aliveFeatures]
        if len(prunedCols) == len(baseColsList) or len(prunedCols) < 2:
            return baseEmb, baseScore, loadingsDf, prunedCols
        baseScaled = prepareManifoldData(dfDim, baseColsList)
        baseR2 = getEmbeddingR2(baseScaled, baseEmb)
        baseAdjR2 = getAdjR2(baseR2, baseScaled.shape[0], len(baseColsList))
        cleanScaled = prepareManifoldData(dfDim, prunedCols)
        try:
            if algoName == 'PCA':
                newEmb = KernelPCA(n_components=2, kernel='cosine', random_state=42).fit_transform(cleanScaled)
            elif algoName == 't-SNE':
                newEmb = TSNE(n_components=2, perplexity=min(30, len(cleanScaled)-1), metric='cosine', init='pca', random_state=42).fit_transform(cleanScaled)
            elif algoName == 'UMAP':
                newEmb = umap.UMAP(n_components=2, metric='cosine', random_state=42).fit_transform(cleanScaled)
            newR2 = getEmbeddingR2(cleanScaled, newEmb)
            newAdjR2 = getAdjR2(newR2, cleanScaled.shape[0], len(prunedCols))
            if newAdjR2 > baseAdjR2:
                newScore = getMultiScaleTrust(cleanScaled, newEmb)
                newLoadings = pd.DataFrame({
                    'Comp 1': getManifoldImportance(cleanScaled, newEmb[:, 0], prunedCols),
                    'Comp 2': getManifoldImportance(cleanScaled, newEmb[:, 1], prunedCols)
                }).sort_values(by='Comp 1', ascending=False)
                return newEmb, newScore, newLoadings, prunedCols
            else:
                return baseEmb, baseScore, loadingsDf, baseColsList
        except:
            return baseEmb, baseScore, loadingsDf, baseColsList
    
    pcaUnsupEmb, pcaUnsupScore, pcaUnsupLoadingsDf, pcaUnsupCols = applyMiPruning(
        pcaUnsupLoadingsDf, uCols, 'PCA', pcaUnsupEmb, pcaUnsupScore)
    
    tsneUnsupEmb, tsneUnsupScore, tsneUnsupLoadingsDf, tsneUnsupCols = applyMiPruning(
        tsneUnsupLoadingsDf, uCols, 't-SNE', tsneUnsupEmb, tsneUnsupScore)
    
    umapUnsupEmb, umapUnsupScore, umapUnsupLoadingsDf, umapUnsupCols = applyMiPruning(
        umapUnsupLoadingsDf, uCols, 'UMAP', umapUnsupEmb, umapUnsupScore)
    
    pcaUnsupLoadingsDf = pcaUnsupLoadingsDf.sort_values(by='Comp 1', ascending=False).head(8)
    tsneUnsupLoadingsDf = tsneUnsupLoadingsDf.sort_values(by='Comp 1', ascending=False).head(8)
    umapUnsupLoadingsDf = umapUnsupLoadingsDf.sort_values(by='Comp 1', ascending=False).head(8)
    
    dfDim['uPCA_1'], dfDim['uPCA_2'] = pcaUnsupEmb[:, 0], pcaUnsupEmb[:, 1]
    dfDim['uTSNE_1'], dfDim['uTSNE_2'] = tsneUnsupEmb[:, 0], tsneUnsupEmb[:, 1]
    dfDim['uUMAP_1'], dfDim['uUMAP_2'] = umapUnsupEmb[:, 0], umapUnsupEmb[:, 1]
    dfDim['PCA_1'], dfDim['PCA_2'] = bestResults['PCA']['emb'][:, 0], bestResults['PCA']['emb'][:, 1]
    dfDim['TSNE_1'], dfDim['TSNE_2'] = bestResults['TSNE']['emb'][:, 0], bestResults['TSNE']['emb'][:, 1]
    dfDim['UMAP_1'], dfDim['UMAP_2'] = bestResults['UMAP']['emb'][:, 0], bestResults['UMAP']['emb'][:, 1]
    dfDim['Shape'] = dfDim['isSquare'].map({True: 'Square', False: 'Rectangular'})
    
    def getFinalR2(cols, embData):
        scaledData = prepareManifoldData(dfDim, cols)
        return getEmbeddingR2(scaledData, embData)
    
    pcaUnsupR2 = getFinalR2(pcaUnsupCols, pcaUnsupEmb)
    tsneUnsupR2 = getFinalR2(tsneUnsupCols, tsneUnsupEmb)
    umapUnsupR2 = getFinalR2(umapUnsupCols, umapUnsupEmb)

joblib.dump({
            'dfDim': dfDim, 'bestResults': bestResults,
            'pcaLoadingsDf': pcaLoadingsDf, 'tsneLoadingsDf': tsneLoadingsDf, 'umapLoadingsDf': umapLoadingsDf,
            'pcaUnsupCols': pcaUnsupCols, 'tsneUnsupCols': tsneUnsupCols, 'umapUnsupCols': umapUnsupCols,
            'pcaUnsupScore': pcaUnsupScore, 'tsneUnsupScore': tsneUnsupScore, 'umapUnsupScore': umapUnsupScore,
            'pcaUnsupLoadingsDf': pcaUnsupLoadingsDf, 'tsneUnsupLoadingsDf': tsneUnsupLoadingsDf, 'umapUnsupLoadingsDf': umapUnsupLoadingsDf,
            'pcaUnsupR2': pcaUnsupR2, 'tsneUnsupR2': tsneUnsupR2, 'umapUnsupR2': umapUnsupR2
        }, dimCheckpointPath)

# ============================
# PLOTTING
# ============================
figDim, axesDim = plt.subplots(4, 3, figsize=(24, 28))

# Row 1: Targeted/Supervised Scatter
sns.scatterplot(data=dfDim, x='PCA_1', y='PCA_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[0, 0], legend=False)
axesDim[0, 0].set_title(f"Targeted PCA ({len(bestResults['PCA']['prunedCols'])} vars)\nVar (R²): {bestResults['PCA']['r2']:.2f} | Trustworthiness: {bestResults['PCA']['score']:.2f}", fontsize=14, fontweight='bold')
sns.scatterplot(data=dfDim, x='TSNE_1', y='TSNE_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[0, 1], legend=False)
axesDim[0, 1].set_title(f"Targeted t-SNE ({len(bestResults['TSNE']['prunedCols'])} vars)\nVar (R²): {bestResults['TSNE']['r2']:.2f} | Trustworthiness: {bestResults['TSNE']['score']:.2f}", fontsize=14, fontweight='bold')
sns.scatterplot(data=dfDim, x='UMAP_1', y='UMAP_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[0, 2], legend=False)
axesDim[0, 2].set_title(f"Fully Supervised UMAP ({len(bestResults['UMAP']['prunedCols'])} vars)\nVar (R²): {bestResults['UMAP']['r2']:.2f} | Trustworthiness: {bestResults['UMAP']['score']:.2f}", fontsize=14, fontweight='bold')

# Row 2: Targeted/Supervised Heatmap
sns.heatmap(pcaLoadingsDf, cmap='viridis', annot=True, fmt=".2f", ax=axesDim[1, 0])
axesDim[1, 0].set_title('Targeted PCA Absolute Loadings', fontsize=12, fontweight='bold')
axesDim[1, 0].set_yticklabels([t.get_text().replace(' ', '\n') for t in axesDim[1, 0].get_yticklabels()], rotation=0)
sns.heatmap(tsneLoadingsDf, cmap='viridis', annot=True, fmt=".2f", ax=axesDim[1, 1])
axesDim[1, 1].set_title('Targeted t-SNE MI Scores', fontsize=12, fontweight='bold')
axesDim[1, 1].set_yticklabels([t.get_text().replace(' ', '\n') for t in axesDim[1, 1].get_yticklabels()], rotation=0)
sns.heatmap(umapLoadingsDf, cmap='viridis', annot=True, fmt=".2f", ax=axesDim[1, 2])
axesDim[1, 2].set_title('Supervised UMAP MI Scores', fontsize=12, fontweight='bold')
axesDim[1, 2].set_yticklabels([t.get_text().replace(' ', '\n') for t in axesDim[1, 2].get_yticklabels()], rotation=0)

# Row 3: Unsupervised Scatter
sns.scatterplot(data=dfDim, x='uPCA_1', y='uPCA_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[2, 0])
axesDim[2, 0].set_title(f"Unsupervised PCA ({len(pcaUnsupCols)} vars)\nVar (R²): {pcaUnsupR2:.2f} | Trustworthiness: {pcaUnsupScore:.2f}", fontsize=14, fontweight='bold')
axesDim[2, 0].legend(loc='upper right', fontsize=10)
sns.scatterplot(data=dfDim, x='uTSNE_1', y='uTSNE_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[2, 1], legend=False)
axesDim[2, 1].set_title(f"Unsupervised t-SNE ({len(tsneUnsupCols)} vars)\nVar (R²): {tsneUnsupR2:.2f} | Trustworthiness: {tsneUnsupScore:.2f}", fontsize=14, fontweight='bold')
sns.scatterplot(data=dfDim, x='uUMAP_1', y='uUMAP_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[2, 2], legend=False)
axesDim[2, 2].set_title(f"Unsupervised UMAP ({len(umapUnsupCols)} vars)\nVar (R²): {umapUnsupR2:.2f} | Trustworthiness: {umapUnsupScore:.2f}", fontsize=14, fontweight='bold')

# Row 4: Unsupervised Heatmaps
sns.heatmap(pcaUnsupLoadingsDf, cmap='viridis', annot=True, fmt=".2f", ax=axesDim[3, 0])
axesDim[3, 0].set_title('Unsupervised PCA Absolute Loadings', fontsize=12, fontweight='bold')
axesDim[3, 0].set_yticklabels([t.get_text().replace(' ', '\n') for t in axesDim[3, 0].get_yticklabels()], rotation=0)
sns.heatmap(tsneUnsupLoadingsDf, cmap='viridis', annot=True, fmt=".2f", ax=axesDim[3, 1])
axesDim[3, 1].set_title('Unsupervised t-SNE MI Scores', fontsize=12, fontweight='bold')
axesDim[3, 1].set_yticklabels([t.get_text().replace(' ', '\n') for t in axesDim[3, 1].get_yticklabels()], rotation=0)
sns.heatmap(umapUnsupLoadingsDf, cmap='viridis', annot=True, fmt=".2f", ax=axesDim[3, 2])
axesDim[3, 2].set_title('Unsupervised UMAP MI Scores', fontsize=12, fontweight='bold')
axesDim[3, 2].set_yticklabels([t.get_text().replace(' ', '\n') for t in axesDim[3, 2].get_yticklabels()], rotation=0)

# ===========================================
# CLUSTERING ON EMBEDDINGS (DBSCAN vs KMeans)
# ===========================================
@cacheDir.cache
def tune_clustering(embData, algoTitle):
    bestScore = -1
    bestLabels = None
    bestName = "None"
    # Test KMeans
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(embData)
        score = silhouette_score(embData, kmeans.labels_)
        if score > bestScore:
            bestScore, bestLabels, bestName = score, kmeans.labels_, f"KMeans (k={k})"
    # Test DBSCAN (Scale first for consistent eps distance)
    scaled_emb = StandardScaler().fit_transform(embData)
    for eps in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        for minSamples in [5, 10, 15]:
            db = DBSCAN(eps=eps, min_samples=minSamples).fit(scaled_emb)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if 1 < n_clusters <= 8:
                valid_mask = labels != -1
                if valid_mask.sum() > len(labels) * 0.5:
                    sc = silhouette_score(scaled_emb[valid_mask], labels[valid_mask])
                    penalized_sc = sc * (valid_mask.sum() / len(labels))
                    if penalized_sc > bestScore:
                        bestScore, bestLabels, bestName = penalized_sc, labels, f"DBSCAN (eps={eps}, ms={minSamples})"
    return bestLabels, bestName, bestScore

clusterTargets = [
    ('Targeted PCA', 'PCA_1', 'PCA_2'),
    ('Targeted t-SNE', 'TSNE_1', 'TSNE_2'),
    ('Supervised UMAP', 'UMAP_1', 'UMAP_2'),
    ('Unsupervised PCA', 'uPCA_1', 'uPCA_2'),
    ('Unsupervised t-SNE', 'uTSNE_1', 'uTSNE_2'),
    ('Unsupervised UMAP', 'uUMAP_1', 'uUMAP_2')
]

# Set up the 4x3 Alternating Grid
figClust, axesClust = plt.subplots(4, 3, figsize=(24, 28))
figClust.suptitle("Optimal Clustering Regions & Literal Matrix 'Kind' Composition", fontsize=20, fontweight='bold')

# Helper function to aggregate verbose matrix types into 7 core scientific domains
def aggregate_matrix_kind(kind):
    if pd.isna(kind): return 'Unknown'
    k = str(kind).lower()
    if 'power network' in k or 'circuit' in k or 'semiconductor' in k or 'electromagnetic' in k: return 'Electrical & Power'
    if 'graph' in k or 'network' in k or 'citation' in k or 'web' in k or 'social' in k: return 'Graphs & Networks'
    if 'optimization' in k or 'linear programming' in k or 'least squares' in k or 'economic' in k: return 'Optimization & LP'
    if 'fluid' in k or 'chemical' in k or 'navier' in k: return 'Fluid & Chemical Dynamics'
    if 'structural' in k or 'thermal' in k or 'material' in k or 'mechanical' in k or 'acoustic' in k: return 'Structural & Physical'
    if '2d' in k or '3d' in k or 'graphics' in k or 'vision' in k: return '2D/3D Spatial'
    return 'Other Mathematical'

for idx, (title, col1, col2) in enumerate(clusterTargets):
    # Math to alternate rows:
    axScat = axesClust[(idx // 3) * 2, idx % 3]
    axBar = axesClust[(idx // 3) * 2 + 1, idx % 3]
    # Extract 2D coords
    embData = dfDim[[col1, col2]].values
    labels, algoName, score = tune_clustering(embData, title)
    dfDim[f'Cluster_{idx}'] = labels
    # Separate noise (-1) from clustered points
    noiseMask = dfDim[f'Cluster_{idx}'] == -1
    clusteredData = dfDim[~noiseMask]
    noiseData = dfDim[noiseMask]
    sns.scatterplot(data=clusteredData, x=col1, y=col2, hue=f'Cluster_{idx}', palette='tab10', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.9, ax=axScat, legend='full')
    if noiseMask.sum() > 0: axScat.scatter(noiseData[col1], noiseData[col2], c='grey', marker='x', s=60, alpha=0.5, label='Noise (-1)')
    axScat.set_title(f"{title}\nAlgorithm: {algoName} | Silhouette: {score:.2f}", fontsize=14, fontweight='bold')
    axScat.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title="Cluster ID")
    # Apply the mapping
    aggKindCol = clusteredData['Kind'].apply(aggregate_matrix_kind)
    # Calculate percentage using the 7 aggregated categories
    compTable = pd.crosstab(clusteredData[f'Cluster_{idx}'], aggKindCol, normalize='index') * 100
    # Plot stacked bar chart (using Set2 for distinct, readable colors)
    compTable.plot(kind='bar', stacked=True, ax=axBar, colormap='Set2', edgecolor='black')
    axBar.set_title("Composition by Matrix Category", fontsize=14, fontweight='bold')
    axBar.set_xlabel("Cluster ID", fontsize=12)
    axBar.set_ylabel("Percentage (%)", fontsize=12)
    axBar.set_xticklabels(axBar.get_xticklabels(), rotation=0)
    axBar.legend(title="Aggregated Category", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# =============================
# DECISION TREES: FAILURE MODES
# =============================

treeFeatures = ['Topological Entropy', 'RCM Bandwidth', 'Diagonally Dominant\nRow Fraction', 'isSquare', 'Density', 'Degeneracy Multiplier', 
                'Signed Frobenius Ratio', 'Brauer Max Product', 'Brauer Ratio']
masterTreeData = dfTransformed.copy()

for col in treeFeatures:
    if col != 'isSquare':
        masterTreeData[col] = np.clip(pd.to_numeric(masterTreeData[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(-1), a_min=-1e35, a_max=1e35)
        
figTrees, axesTrees = plt.subplots(nrows=5, ncols=1, figsize=(30, 60))
figTrees.suptitle("Decision Trees: Topological Rules for Matrix Constraints", fontsize=24, fontweight='bold')
axesTrees = axesTrees.flatten()

paramGrid = {
    'max_depth': [3, 4, 5],
    'max_leaf_nodes': [3, 4, 5, 6],
    'min_samples_split': [0.10, 0.15],
    'min_samples_leaf': [0.05, 0.075],
    'criterion': ['gini', 'entropy']
}

# Iterate, Optimize, and Plot each Target
for idx, targetCol in enumerate(binaryTargets):
    currAx = axesTrees[idx]
    targetData = masterTreeData.copy()
    if targetCol != 'isSvdFailed':
        if targetData[targetCol].dtype == object:
            targetData[targetCol] = targetData[targetCol].replace({'yes': 1, 'Yes': 1, '1': 1, 'no': 0, 'No': 0, '0': 0})
        targetData[targetCol] = pd.to_numeric(targetData[targetCol], errors='coerce')
        targetData = targetData.dropna(subset=[targetCol])

    if len(targetData) < 20:
        currAx.set_axis_off()
        currAx.set_title(f"Target: {targetCol}\n(Insufficient Data)", fontsize=16)
        continue

    xTree = targetData[treeFeatures].copy()
    yTree = (targetData[targetCol] > 0).astype(int)
    # Ensure there are at least two classes to split
    if len(np.unique(yTree)) < 2:
        currAx.set_axis_off()
        currAx.set_title(f"Target: {targetCol}\n(Zero Target Variance)", fontsize=16, color='maroon')
        continue

    # Grid Search for the mathematically optimal tree structure
    @cacheDir.cache
    def get_optimal_failure_tree(xData, yData, targetName):
        baseTree = DecisionTreeClassifier(random_state=42)
        mcCv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
        gridSearch = GridSearchCV(baseTree, paramGrid, cv=mcCv, scoring='balanced_accuracy', n_jobs=-1)
        gridSearch.fit(xData, yData)
        return gridSearch.best_estimator_, gridSearch.best_score_

    bestTree, bestCvScore = get_optimal_failure_tree(xTree, yTree, targetCol)
    if targetCol == 'isSvdFailed':
        classNames = ['SVD Success', 'SVD Failed']
    else:
        classNames = [f'Not {targetCol}', f'Is {targetCol}']

    plot_tree(bestTree, feature_names=treeFeatures, class_names=classNames, filled=True, rounded=True, ax=currAx, fontsize=10, proportion=False, impurity=False)
    currAx.set_title(f"Target: {targetCol}\nCV Balanced Acc: {bestCvScore:.3f} | Depth: {bestTree.max_depth}", fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# ============================================
# COMPOSITE TREE CLUSTERING & MANIFOLD OVERLAY
# ============================================

treeFeatures = ['Topological Entropy', 'RCM Bandwidth', 'Diagonally Dominant\nRow Fraction', 'isSquare', 'Density', 'Degeneracy Multiplier', 'Signed Frobenius Ratio',
                'Signed Frobenius Ratio', 'Brauer Max Product', 'Brauer Ratio']
binaryTargets = ['isSvdFailed', 'Rank Collapse', 'Positive Definite', 'isCholesky', 'isIrreducible']

# 2. Dynamically evaluate and pool independent top predictors
dynamicFeatures = set()
basePredictors = [c for c in masterPredictors if c in dfTransformed.columns and c != 'isSquare']

for targetCol in binaryTargets:
    evalSubset = dfTransformed.dropna(subset=[targetCol] + basePredictors).copy()
    if len(evalSubset) < 5: continue
    yEval = (evalSubset[targetCol] > 0).astype(int)
    if len(np.unique(yEval)) < 2: continue
    xEval = evalSubset[basePredictors].replace([np.inf, -np.inf], [1e35, -1e35])
    targetCorr = xEval.corrwith(pd.Series(yEval, index=xEval.index), method='spearman').abs().fillna(0)
    nNeighbors = min(3, len(xEval) - 1)
    xEvalImp = SimpleImputer(strategy='median', add_indicator=True).fit_transform(xEval)
    miScores = mutual_info_classif(xEvalImp, yEval, n_neighbors=nNeighbors, random_state=42)
    miSeries = pd.Series(miScores[:len(basePredictors)], index=basePredictors).sort_values(ascending=False)
    corrMatrix = xEval.corr(method='spearman').abs()
    featuresToDrop = set()
    for i in range(len(corrMatrix.columns)):
        for j in range(i + 1, len(corrMatrix.columns)):
            colA = corrMatrix.columns[i]
            colB = corrMatrix.columns[j]
            if corrMatrix.iloc[i, j] > 0.85:
                if targetCorr[colA] > targetCorr[colB]:
                    featuresToDrop.add(colB)
                else:
                    featuresToDrop.add(colA)
    survivingFeatures = [f for f in miSeries.index if f not in featuresToDrop]
    topCount = 0
    for feat in survivingFeatures:
        dynamicFeatures.add(feat)
        topCount += 1
        if topCount >= 2: break
treeFeatures = list(dynamicFeatures)
if 'isSquare' not in treeFeatures:
    treeFeatures.append('isSquare')

# 3. Proceed with evalData generation and tree training
evalData = dfDim.dropna(subset=treeFeatures + ['uUMAP_1', 'uUMAP_2', 'uTSNE_1', 'uTSNE_2', 'uPCA_1', 'uPCA_2']).copy()
for col in treeFeatures:
    if col != 'isSquare':
        evalData[col] = pd.to_numeric(evalData[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(-1)
evalData['isSquare'] = evalData['isSquare'].astype(int)
evalData['isSvdFailed'] = (evalData['Condition Number'].isna() | evalData['Minimum Singular Value'].isna()).astype(int)
for t in ['isSvdFailed', 'isIrreducible', 'Rank Collapse', 'isCholesky']:
    if t in dfTransformed.columns:
        evalData[t] = dfTransformed.loc[evalData.index, t]
trainedTrees = {}
paramGrid = {
    'max_depth': [3, 4, 5],
    'max_leaf_nodes': [3, 4, 6, 6, 7],
    'min_samples_split': [0.10, 0.15],
    'min_samples_leaf': [0.05, 0.075],
    'criterion': ['gini', 'entropy']
}

for targetCol in binaryTargets:
    targetData = evalData.copy()
    if targetCol != 'isSvdFailed':
        targetData[targetCol] = pd.to_numeric(targetData[targetCol], errors='coerce')
        targetData = targetData.dropna(subset=[targetCol])
    xTree = targetData[treeFeatures].copy()
    yTree = (targetData[targetCol] > 0).astype(int)
    # If the target has no variance, skip training and store None
    if len(np.unique(yTree)) < 2:
        trainedTrees[targetCol] = None
        continue
    
    @cacheDir.cache
    def get_optimal_composite_tree(xData, yData, targetName):
        baseTree = DecisionTreeClassifier(random_state=42)
        gridSearch = GridSearchCV(baseTree, paramGrid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
        gridSearch.fit(xData, yData)
        return gridSearch.best_estimator_
    trainedTrees[targetCol] = get_optimal_composite_tree(xTree, yTree, targetCol)

# Predict the constraints across the full manifold space
for targetCol, model in trainedTrees.items():
    predColName = targetCol + 'Pred'
    if model is not None:
        evalData[predColName] = model.predict(evalData[treeFeatures])
    else:
        # Default to 0 (No flag) if the model couldn't be trained due to zero variance
        evalData[predColName] = 0

def buildProfile(row):
    flags = []
    if row.get('isSvdFailedPred', 0) == 1: flags.append('SVD Fail')
    if row.get('Rank CollapsePred', 0) == 1: flags.append('Rank Collapse')
    if row.get('Positive DefinitePred', 0) == 1: flags.append('Pos Definite')
    if row.get('isCholeskyPred', 0) == 1: flags.append('Cholesky')
    if row.get('isIrreduciblePred', 0) == 1: flags.append('Irreducible')
    if len(flags) == 0: return 'Standard (No Flags)'
    return " + ".join(flags)

evalData['compositeProfile'] = evalData.apply(buildProfile, axis=1)

# Build the Composite State (The Deterministic Cluster)
constraintColors = {
    'isSvdFailedPred': 'magenta',
    'Rank CollapsePred': 'cyan',
    'Positive DefinitePred': 'red',
    'isCholeskyPred': 'yellow',
    'isIrreduciblePred': 'lime'
}

# Bulletproof sumFlags calculation: maps directly from base targets and verifies existence
validPredCols = [targetCol + 'Pred' for targetCol in binaryTargets if targetCol + 'Pred' in evalData.columns]
evalData['sumFlags'] = evalData[validPredCols].sum(axis=1)
  
for rawCol in ['Rank', 'Condition Number', 'Full Numerical Rank?']:
    if rawCol in dfTransformed.columns:
        evalData[rawCol] = dfTransformed.loc[evalData.index, rawCol]

# Group the targets for visual isolation
oppTargets = ['Positive Definite', 'isIrreducible', 'isCholesky']
failTargets = ['isSvdFailed', 'Rank Collapse']

# Expand to a 3x3 grid
figComp, axesComp = plt.subplots(3, 3, figsize=(26, 24))
figComp.suptitle("Composite Decision Tree Predictions & Uncomputable Matrices", fontsize=22, fontweight='bold')

manifolds = [('PCA', 'uPCA_1', 'uPCA_2'), ('t-SNE', 'uTSNE_1', 'uTSNE_2'), ('UMAP', 'uUMAP_1', 'uUMAP_2')]

for idx, (mName, xCol, yCol) in enumerate(manifolds):
    axTop = axesComp[0, idx]
    baseMask = evalData['sumFlags'] == 0
    sns.scatterplot(data=evalData[baseMask], x=xCol, y=yCol, color='lightgrey', style='isSquare', markers={1: 's', 0: 'o'}, s=80, alpha=0.3, ax=axTop, label='Standard (No Flags)', legend=False)
    for targetCol in oppTargets:
        if targetCol not in trainedTrees or trainedTrees[targetCol] is None: continue
        predCol = targetCol + 'Pred'
        if predCol in evalData.columns:
            mask = evalData[predCol] == 1
            if mask.sum() > 0:
                color = constraintColors.get(predCol, 'black')
                sns.scatterplot(data=evalData[mask], x=xCol, y=yCol, color=color, style='isSquare', markers={1: 's', 0: 'o'}, s=80, alpha=0.4, ax=axTop, label=targetCol, legend=False)
    axTop.set_title(f"Unsupervised {mName}\n(Decomposition / Opportunity Modes)", fontsize=14, fontweight='bold')
    axTop.set_ylabel(f"{mName} Dimension 2")
    axTop.grid(True, linestyle=':', alpha=0.5)
    if idx == 0:
        handles, labels = axTop.get_legend_handles_labels()
        byLabel = dict(zip(labels, handles))
        axTop.legend(byLabel.values(), byLabel.keys(), loc='upper right', fontsize=11, title="Opportunity Layers")
    axMid = axesComp[1, idx]
    sns.scatterplot(data=evalData[baseMask], x=xCol, y=yCol, color='lightgrey', style='isSquare', markers={1: 's', 0: 'o'}, s=80, alpha=0.3, ax=axMid, label='Standard (No Flags)', legend=False)
    for targetCol in failTargets:
        if targetCol not in trainedTrees or trainedTrees[targetCol] is None: continue
        predCol = targetCol + 'Pred'
        if predCol in evalData.columns:
            mask = evalData[predCol] == 1
            if mask.sum() > 0:
                color = constraintColors.get(predCol, 'black')
                sns.scatterplot(data=evalData[mask], x=xCol, y=yCol, color=color, style='isSquare', markers={1: 's', 0: 'o'}, s=80, alpha=0.6, ax=axMid, label=targetCol, legend=False)
    axMid.set_title(f"Unsupervised {mName}\n(Predicted Failure Modes)", fontsize=14, fontweight='bold')
    axMid.set_ylabel(f"{mName} Dimension 2")
    axMid.grid(True, linestyle=':', alpha=0.5)
    if idx == 0:
        handles, labels = axMid.get_legend_handles_labels()
        byLabel = dict(zip(labels, handles))
        axMid.legend(byLabel.values(), byLabel.keys(), loc='upper right', fontsize=11, title="Failure Layers")
    axBot = axesComp[2, idx]
    # Identifies matrices missing critical Rank or Condition Number values
    rankMissing = evalData['Rank'].isna() if 'Rank' in evalData.columns else evalData['Full Numerical Rank?'].isna()
    condMissing = evalData['Condition Number'].isna()
    computeFailMask = rankMissing | condMissing
    computeSuccessMask = ~computeFailMask
    axBot.scatter(evalData.loc[computeSuccessMask, xCol], evalData.loc[computeSuccessMask, yCol], c='lightgrey', alpha=0.3, s=50, label='Successfully Computed')
    failCount = computeFailMask.sum()
    if failCount > 0:
        axBot.scatter(evalData.loc[computeFailMask, xCol], evalData.loc[computeFailMask, yCol], c='red', marker='*', s=200, edgecolor='black', linewidth=1, alpha=0.8, label='Failed to Compute')
    axBot.set_title(f"Unsupervised {mName} Projection\n({failCount} Uncomputable Matrices)", fontsize=14, fontweight='bold')
    axBot.set_xlabel(f"{mName} Dimension 1")
    axBot.set_ylabel(f"{mName} Dimension 2")
    axBot.grid(True, linestyle=':', alpha=0.5)
    if idx == 0:
        axBot.legend(loc='upper right', fontsize=11)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# ====================================
# STATISTICAL VERIFICATION OF CLUSTERS
# ====================================

# Merge the profiles back into the master transformed dataset
dfTransformed['compositeProfile'] = evalData['compositeProfile']
dfTransformed['compositeProfile'] = dfTransformed['compositeProfile'].fillna('Unknown')

# Lock in the top 4 most common distinct profiles
lockedProfiles = [p for p in evalData['compositeProfile'].value_counts().index if p != 'Unknown'][:4]
expertFeatures = [c for c in xgbFeatures if c not in treeFeatures]
if 'isInfinitePositivity' in dfTransformed.columns and 'isInfinitePositivity' not in expertFeatures:
    expertFeatures.append('isInfinitePositivity')

# Kruskal-Wallis Test: Do the expert features differ across the locked profiles?
verificationResults = []
for feature in expertFeatures:
    groupData = []
    for profile in lockedProfiles:
        data = dfTransformed[dfTransformed['compositeProfile'] == profile][feature].dropna()
        if len(data) > 5:
            groupData.append(data)
    if len(groupData) >= 2:
        try:
            stat, pVal = stats.kruskal(*groupData)
            verificationResults.append({'Feature': feature, 'p-value': pVal})
        except: pass
dfVerification = pd.DataFrame(verificationResults)
dfVerification['Statistically Distinct'] = dfVerification['p-value'] < 0.05

# ==============================================
# STATISTICAL VERIFICATION (HEATMAP & LINEARITY)
# ==============================================
matrixTargets = [
    'Condition Number', 'Matrix Norm', 'isSvdFailed',
    'Rank Collapse', 'isIrreducible', 'isCholesky', 'Positive Definite'
]

profileCounts = evalData['compositeProfile'].value_counts()
lockedProfiles = [p for p in profileCounts.index if p != 'Unknown' and profileCounts[p] >= 5]
finalResults = []

for tIdx, targetCol in enumerate(matrixTargets):
    for pIdx, profileName in enumerate(lockedProfiles):
        localDf = dfTransformed[dfTransformed['compositeProfile'] == profileName].copy()
        testData = localDf[[targetCol] + expertFeatures].copy()
        if targetCol in ['Condition Number', 'Matrix Norm']:
            testData[targetCol] = pd.to_numeric(testData[targetCol], errors='coerce').replace([np.inf, -np.inf], np.nan)
        else:
            if testData[targetCol].dtype == object:
                testData[targetCol] = testData[targetCol].replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0})
            testData[targetCol] = pd.to_numeric(testData[targetCol], errors='coerce')
        testData = testData.dropna(subset=[targetCol])

        if len(testData) < 5:
            finalResults.append({'Target': targetCol, 'Profile': profileName, 'Top Predictor': "Insufficient Data", 'MI Score': 0.0, 'Linear PR-AUC': 0.0, 'Linear R²': 0.0})
            continue

        xStat = testData[expertFeatures].copy()
        yStat = testData[targetCol]
        isClassification = targetCol in ['Rank Collapse', 'Positive Definite', 'isSvdFailed', 'isIrreducible', 'isCholesky']
        for colName in expertFeatures:
            if colName in ['Minimum Singular Value', 'Density', 'Diagonally Dominant\nRow Fraction', 'RCM Compression Ratio']:
                xStat[colName] = xStat[colName].fillna(0.0)
            else:
                xStat[colName] = xStat[colName].fillna(1e35)
            safeCol = pd.to_numeric(xStat[colName], errors='coerce').astype(float)
            xStat[colName] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
        if isClassification:
            binaryY = (yStat > 0).astype(int)
            if len(np.unique(binaryY)) < 2:
                ruleText = "Absolute Law\n(100% True)" if binaryY.iloc[0] == 1 else "Absolute Law\n(100% False)"

                finalResults.append({'Target': targetCol, 'Profile': profileName, 'Top Predictor': ruleText, 'MI Score': 0.0, 'Linear PR-AUC': 0.0, 'Linear R²': 0.0})
                continue
            targetCorr = xStat.corrwith(pd.Series(binaryY, index=xStat.index), method='spearman').abs()
        else:
            if yStat.nunique() < 2:
                finalResults.append({'Target': targetCol, 'Profile': profileName, 'Top Predictor': "Constant Value\n(No Variance)", 'MI Score': 0.0, 'Linear PR-AUC': 0.0, 'Linear R²': 0.0})
                continue
            targetCorr = xStat.corrwith(yStat, method='spearman').abs()
        corrMatrix = xStat.corr(method='spearman').abs()
        featuresToDrop = set()
        for i in range(len(corrMatrix.columns)):
            for j in range(i + 1, len(corrMatrix.columns)):
                colA = corrMatrix.columns[i]
                colB = corrMatrix.columns[j]
                if corrMatrix.iloc[i, j] > 0.85:
                    if targetCorr[colA] > targetCorr[colB]:
                        featuresToDrop.add(colB)
                    else:
                        featuresToDrop.add(colA)
        xFiltered = xStat.drop(columns=list(featuresToDrop)).replace([np.inf, -np.inf], [1e35, -1e35])
        if xFiltered.shape[1] == 0:
            finalResults.append({'Target': targetCol, 'Profile': profileName, 'Top Predictor': "Features Collinear", 'MI Score': 0.0, 'Linear PR-AUC': 0.0, 'Linear R²': 0.0})
            continue
        linPrAuc, linR2 = np.nan, np.nan
        treePrAuc, treeR2 = np.nan, np.nan
        # Dynamically scale n_neighbors to prevent the ValueError
        nNeighbors = min(3, len(xFiltered) - 1)
        xFilteredImp = SimpleImputer(strategy='median', add_indicator=True).fit_transform(xFiltered)
        if isClassification:
            miScores = mutual_info_classif(xFilteredImp, binaryY, n_neighbors=nNeighbors, random_state=42)
            linPipelineClassif = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='saga', C=0.1, random_state=42, max_iter=5000))
            try:
                yPredProbLin = cross_val_predict(linPipelineClassif, xFiltered, binaryY, cv=5, method='predict_proba')[:, 1]
                linPrAuc = average_precision_score(binaryY, yPredProbLin)
            except: pass
            posCount = sum(binaryY)
            posWeight = (len(binaryY) - posCount) / posCount if posCount > 0 else 1
            treeModel = xgb.XGBClassifier(n_estimators=50, max_depth=3, scale_pos_weight=posWeight, random_state=42, eval_metric='logloss')
            try:
                yPredProbTree = cross_val_predict(treeModel, xFiltered, binaryY, cv=5, method='predict_proba')[:, 1]
                treePrAuc = average_precision_score(binaryY, yPredProbTree)
            except: pass
        else:
            if targetCol in ['Condition Number', 'Matrix Norm']:
                regY = np.log1p(np.clip(yStat.values, 0, 1e35))
                miScores = mutual_info_regression(xFilteredImp, regY, n_neighbors=nNeighbors, random_state=42)
                linPipelineReg = make_pipeline(StandardScaler(), Lasso(alpha=0.05, random_state=42, max_iter=10000))
                try:
                    yPredLin = cross_val_predict(linPipelineReg, xFiltered, regY, cv=5)
                    linR2 = r2_score(regY, yPredLin)
                except: pass
                treeModel = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
                try:
                    yPredTree = cross_val_predict(treeModel, xFiltered, regY, cv=5)
                    treeR2 = r2_score(regY, yPredTree)
                except: pass
            else:
                qt = QuantileTransformer(output_distribution='uniform', random_state=42)
                qtY = qt.fit_transform(yStat.values.reshape(-1, 1)).ravel()
                miScores = mutual_info_regression(xFilteredImp, qtY, n_neighbors=nNeighbors, random_state=42)
                linPipelineReg = make_pipeline(StandardScaler(), Lasso(alpha=0.1, random_state=42))
                try:
                    yPredLin = cross_val_predict(linPipelineReg, xFiltered, qtY, cv=5)
                    linR2 = r2_score(qtY, yPredLin)
                except: pass
                treeModel = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
                try:
                    yPredTree = cross_val_predict(treeModel, xFiltered, qtY, cv=5)
                    treeR2 = r2_score(qtY, yPredTree)
                except: pass
        miSeries = pd.Series(miScores[:len(xFiltered.columns)], index=xFiltered.columns).sort_values(ascending=False)
        finalResults.append({
            'Target': targetCol,
            'Profile': profileName,
            'Top Predictor': miSeries.index[0],
            'MI Score': miSeries.iloc[0],
            'Linear PR-AUC': linPrAuc,
            'Linear R²': linR2,
            'Tree PR-AUC': treePrAuc,
            'Tree R²': treeR2
        })

dfProof = pd.DataFrame(finalResults)
dfProof['Max Linear'] = dfProof[['Linear PR-AUC', 'Linear R²']].max(axis=1).fillna(0)
dfProof['Max Tree'] = dfProof[['Tree PR-AUC', 'Tree R²']].max(axis=1).fillna(0)
dfProof['Non-Linear Gap'] = dfProof['Max Tree'] - dfProof['Max Linear']
dfGap = dfProof.dropna(subset=['Non-Linear Gap']).sort_values('Non-Linear Gap', ascending=False).head(15).copy().reset_index(drop=True)

def cleanLabel(row):
    predRaw = str(row['Top Predictor']).split('\n')[0].split('Name:')[0]
    targRaw = str(row['Target']).split('\n')[0].split('Name:')[0]
    predClean = re.sub(r'\(.*?\)', '', predRaw)
    predClean = re.sub(r'^[0-9\s\.]+', '', predClean).strip()
    targClean = re.sub(r'\(.*?\)', '', targRaw)
    targClean = re.sub(r'^[0-9\s\.]+', '', targClean).strip()
    predFinal = predClean.replace(' ', '\n')
    return f"{predFinal}\n({targClean})"

dfGap['Gap Label'] = dfGap.apply(cleanLabel, axis=1)

# ==============================
# LINEAR VS NON-LINEAR DOMINANCE
# ==============================

# Create a lookup dictionary mapping (Target, Profile) to a boolean: True if Linear, False if Tree
optimalModelRouter = {}
for _, row in dfProof.iterrows():
    target = row['Target']
    profile = row['Profile']
    # If the gap is highly negative, Linear is dominant. Otherwise, default to the Non-Linear Tree.
    gap = row['Non-Linear Gap']
    isLinearDominant = True if (pd.notna(gap) and gap < -0.02) else False
    optimalModelRouter[(target, profile)] = isLinearDominant
    
dfReg = dfProof[dfProof['Linear R²'].notna() & dfProof['Tree R²'].notna()].copy()
dfCls = dfProof[dfProof['Linear PR-AUC'].notna() & dfProof['Tree PR-AUC'].notna()].copy()

cleanLabels = []
for rankIdx, rowDict in enumerate(dfGap.to_dict('records')):
    pred = str(rowDict['Top Predictor']).strip().replace('Product (', '\nProduct (').replace('n C', 'n\nC')
    pred = pred.replace('Brauer Max', 'Brauer Max\n')
    targ = str(rowDict['Target']).strip()
    cleanLabels.append(f"{pred}\n[{targ}]")
dfGap['Gap Label'] = cleanLabels

figProof = plt.figure(figsize=(30, 12))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)

# PANEL 1: Regression (R² vs R²)
axReg = figProof.add_subplot(gs[0])
sns.scatterplot(data=dfReg, x='Linear R²', y='Tree R²', hue='Target', style='Profile', s=200, palette='tab10', ax=axReg)
minPlotBound = max(-1.0, min(dfReg['Linear R²'].min(), dfReg['Tree R²'].min()) - 0.1)
maxPlotBound = max(dfReg['Linear R²'].max(), dfReg['Tree R²'].max()) + 0.05
axReg.set_xlim(minPlotBound, maxPlotBound)
axReg.set_ylim(minPlotBound, maxPlotBound)
axReg.plot([minPlotBound, maxPlotBound], [minPlotBound, maxPlotBound], 'k--', alpha=0.5, label='Linear Boundary (y=x)')
axReg.fill_between([minPlotBound, maxPlotBound], [minPlotBound, maxPlotBound], maxPlotBound, color='blue', alpha=0.05)
axReg.set_title("Regression: Linear vs Non-Linear Predictability", fontsize=16, fontweight='bold')
axReg.set_xlabel("Linear Baseline (Lasso R²)", fontsize=12)
axReg.set_ylabel("Non-Linear Signal (XGBoost R²)", fontsize=12)
axReg.legend(loc='upper left', fontsize=10)
axReg.grid(True, linestyle=':', alpha=0.6)

# PANEL 2: Classification (PR-AUC vs PR-AUC)
axCls = figProof.add_subplot(gs[1])
sns.scatterplot(data=dfCls, x='Linear PR-AUC', y='Tree PR-AUC', hue='Target', style='Profile', s=200, palette='Set2', ax=axCls)
maxVal = max(dfCls['Linear PR-AUC'].max(), dfCls['Tree PR-AUC'].max()) + 0.05
minVal = min(dfCls['Linear PR-AUC'].min(), dfCls['Tree PR-AUC'].min()) - 0.05
axCls.plot([minVal, maxVal], [minVal, maxVal], 'k--', alpha=0.5, label='Linear Boundary (y=x)')
axCls.fill_between([minVal, maxVal], [minVal, maxVal], maxVal, color='blue', alpha=0.05)
axCls.set_title("Classification: Linear vs Non-Linear Predictability", fontsize=16, fontweight='bold')
axCls.set_xlabel("Linear Baseline (L1 Logistic PR-AUC)", fontsize=12)
axCls.set_ylabel("Non-Linear Signal (XGBoost PR-AUC)", fontsize=12)
axCls.legend(loc='lower right', fontsize=10)
axCls.grid(True, linestyle=':', alpha=0.6)

# PANEL 3: Non-Linearity Gap Bar Chart
axGap = figProof.add_subplot(gs[2])
dfGap['Plot Gap'] = dfGap['Non-Linear Gap'].clip(upper=1000)
sns.barplot(data=dfGap, x='Plot Gap', y='Gap Label', hue='Target', dodge=False, palette='viridis', ax=axGap)
axGap.set_xscale('symlog', linthresh=0.01)
for i, row in dfGap.reset_index(drop=True).iterrows():
    trueVal = row['Non-Linear Gap']
    plotVal = row['Plot Gap']
    labelStr = f"{trueVal:.2g}"
    if plotVal >= 0: axGap.text(8e-3, i, f"{labelStr} ", color='white', ha='right', va='center', fontweight='bold', fontsize=10)
axGap.set_title("Top Features Requiring Non-Linear Models", fontsize=16, fontweight='bold')
axGap.set_xlabel("Non-Linearity Gap (Symlog Scale)", fontsize=12)
axGap.set_ylabel("")
axGap.legend(loc='lower right', fontsize=10)
axGap.tick_params(axis='y', labelsize=9)
axGap.grid(True, axis='x', linestyle=':', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ========================================
# THE MASTER PROOF MATRIX
# ========================================
figMatrix, axMatrix = plt.subplots(figsize=(50, 25))
targetOrder = [
    'Positive Definite', 'isIrreducible', 'isCholesky',
    'isSvdFailed', 'Rank Collapse', 'isNonCholesky',
    'Condition Number', 'Matrix Norm'
]

dfProof['Non-Linear Gap'] = dfProof['Max Tree'] - dfProof['Max Linear']
gapPivot = dfProof.pivot(index='Target', columns='Profile', values='Non-Linear Gap')
annotMatrix = dfProof.pivot(index='Target', columns='Profile', values='Top Predictor')
validTargets = [t for t in targetOrder if t in gapPivot.index]
gapPivot = gapPivot.reindex(validTargets)
annotMatrix = annotMatrix.reindex(validTargets)

for col in gapPivot.columns:
    for idx in gapPivot.index:
        mask = (dfProof['Target'] == idx) & (dfProof['Profile'] == col)
        if mask.any():
            row = dfProof[mask].iloc[0]
            if "Homogeneous" in row['Top Predictor'] or "Insufficient" in row['Top Predictor'] or "Collinear" in row['Top Predictor'] or "Law" in row['Top Predictor'] or "Constant" in row['Top Predictor']:
                annotMatrix.at[idx, col] = row['Top Predictor']
                gapPivot.at[idx, col] = 0.0
            else:
                gap = row['Non-Linear Gap']
                if gap < -0.02:
                    winner = "LINEAR DOMINANT"
                elif gap > 0.02:
                    winner = "TREE DOMINANT"
                else:
                    winner = "TIE (USE LINEAR)"
                if abs(gap) > 1000:
                    gapStr = f"{gap:+.2e}"
                else:
                    gapStr = f"{gap:+.2f}"
                annotMatrix.at[idx, col] = f"{row['Top Predictor']}\nGap: {gapStr}\n[{winner}]"
        else:
            annotMatrix.at[idx, col] = "N/A"
            gapPivot.at[idx, col] = 0.0
sns.heatmap(gapPivot, annot=annotMatrix, fmt="", cmap='coolwarm', center=0, vmin=-1.0, vmax=1.0,
            linewidths=2, linecolor='black', ax=axMatrix, annot_kws={"size": 10, "weight": "bold"})
for textObj in axMatrix.texts:
    if "Law" in textObj.get_text():
        textObj.set_color('white')
        x, y = textObj.get_position()
        rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, fill=True, color='#303030', zorder=2)
        axMatrix.add_patch(rect)
        textObj.set_zorder(3)
axMatrix.set_title("Master Proof Matrix: Non-Linear vs Linear Dominance\n(Opportunities & Failures by Topological Profile)", fontsize=24, fontweight='bold')
axMatrix.set_xlabel("Topological Manifold Profiles (Clusters)", fontsize=14)
axMatrix.set_ylabel("Target Conditions (Grouped)", fontsize=14)
xLabels = [label.get_text().replace(' + ', ' +\n') for label in axMatrix.get_xticklabels()]
axMatrix.set_xticklabels(xLabels, rotation=0, ha='center', fontsize=18, weight='bold')
axMatrix.set_yticklabels(axMatrix.get_yticklabels(), rotation=0, fontsize=18, weight='bold')
bottomLim, topLim = axMatrix.get_ylim()
axMatrix.set_ylim(bottomLim + 0.5, topLim - 0.5)

plt.tight_layout()
plt.show()

# =========================================
# LOCAL EXPERTS: XGBOOST & SHAP PER CLUSTER
# =========================================
expertTargets = ['Condition Number', 'Matrix Norm', 'Rank Collapse', 'isSvdFailed', 'isIrreducible', 'isCholesky', 'Positive Definite']
validShapPlots = []

# COMPUTE AND FILTER SIGNIFICANT MODELS
for targetCol in expertTargets:
    for profileName in lockedProfiles:
        localDf = dfTransformed[dfTransformed['compositeProfile'] == profileName].copy()
        xgbData = localDf[[targetCol] + expertFeatures].copy()
        if targetCol in ['Condition Number', 'Matrix Norm']:
            xgbData[targetCol] = xgbData[targetCol].replace([np.inf, -np.inf], np.nan).fillna(1e35)
        elif targetCol == 'Rank Collapse':
            xgbData[targetCol] = xgbData[targetCol].replace([np.inf, -np.inf], np.nan)
        if 'isInfinitePositivity' in xgbData.columns:
            xgbData['isInfinitePositivity'] = xgbData['isInfinitePositivity'].astype(int)
        if targetCol not in ['isInfinitePositivity']:
            xgbData[targetCol] = pd.to_numeric(xgbData[targetCol], errors='coerce').astype(float)
        for col in expertFeatures:
            if col != 'isInfinitePositivity':
                safeCol = pd.to_numeric(xgbData[col], errors='coerce').astype(float)
                xgbData[col] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
        xgbData[expertFeatures] = xgbData[expertFeatures].replace([np.inf, -np.inf], np.nan)
        xgbData = xgbData.dropna(subset=[targetCol])
        if len(xgbData) < 15: continue
        xXgbRaw = xgbData[expertFeatures]
        yXgbRaw = xgbData[targetCol]
        cleanX = xXgbRaw.copy()
        for colName in expertFeatures:
            if colName in ['Minimum Singular Value', 'Density', 'Diagonally Dominant\nRow Fraction', 'RCM Compression Ratio']:
                cleanX[colName] = cleanX[colName].replace([np.inf, -np.inf], np.nan)
            else:
                cleanX[colName] = cleanX[colName].replace([np.inf, -np.inf], np.nan)
            safeCol = pd.to_numeric(cleanX[colName], errors='coerce').astype(float)
            cleanX[colName] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
        isClassification = targetCol in ['Rank Collapse', 'Positive Definite', 'Cholesky Candidate', 'isSvdFailed', 'isIrreducible', 'isCholesky']
        if isClassification:
            yXgb = (yXgbRaw > 0).astype(int)
            posCount = sum(yXgb)
            negCount = len(yXgb) - posCount
            if posCount < 3 or negCount < 3: continue
            targetCorr = cleanX.corrwith(pd.Series(yXgb, index=cleanX.index), method='spearman').abs()
        else:
            if targetCol in ['Condition Number', 'Matrix Norm']: yXgb = np.log1p(np.clip(yXgbRaw, 0, 1e35))
            else: yXgb = yXgbRaw
            if yXgb.std() == 0: continue
            targetCorr = cleanX.corrwith(pd.Series(yXgb, index=cleanX.index), method='spearman').abs()
        corrMatrix = cleanX.corr(method='spearman').abs()
        featuresToDrop = set()
        for i in range(len(corrMatrix.columns)):
            for j in range(i+1, len(corrMatrix.columns)):
                colA, colB = corrMatrix.columns[i], corrMatrix.columns[j]
                if corrMatrix.iloc[i, j] > 0.85:
                    if targetCorr[colA] > targetCorr[colB]: featuresToDrop.add(colB)
                    else: featuresToDrop.add(colA)
        prunedExpertFeatures = [f for f in expertFeatures if f not in featuresToDrop]
        xXgb = cleanX[prunedExpertFeatures]
        isLinear = optimalModelRouter.get((targetCol, profileName), False)
        if isClassification:
            posWeight = negCount / posCount if posCount > 0 else 1
            if isLinear: model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(penalty='l1', solver='saga', C=0.1, random_state=42, max_iter=5000, class_weight='balanced'))
            else: model = xgb.XGBClassifier(n_estimators=75, max_depth=4, learning_rate=0.1, random_state=42, scale_pos_weight=posWeight)
        else:
            if isLinear: model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), Lasso(alpha=0.05, random_state=42, max_iter=10000))
            else: model = xgb.XGBRegressor(n_estimators=75, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(xXgb, yXgb)
        if isLinear:
            activeColumns = model.named_steps['simpleimputer'].get_feature_names_out(xXgb.columns)
            stepName = 'logisticregression' if isClassification else 'lasso'
            fittedLinear = model.named_steps[stepName]
            imputedXXgb = model.named_steps['simpleimputer'].transform(xXgb)
            scaledXXgb = model.named_steps['standardscaler'].transform(imputedXXgb)
            explainer = shap.LinearExplainer(fittedLinear, scaledXXgb)
            shapValues = explainer.shap_values(scaledXXgb)
            displayX = pd.DataFrame(scaledXXgb, columns=activeColumns)
            algoTag = "[L1 Linear]"
        else:
            activeColumns = xXgb.columns
            explainer = shap.TreeExplainer(model)
            shapValues = explainer.shap_values(xXgb)
            displayX = xXgb
            algoTag = "[XGBoost]"
        if isinstance(shapValues, list): shapValues = shapValues[1] if len(shapValues) > 1 else shapValues[0]
        if len(shapValues.shape) == 3: shapValues = shapValues[:, :, 1]
        stdDev = np.std(shapValues, axis=0)
        tMean = np.mean(shapValues, axis=0)
        sigFigs = np.sum(stdDev/tMean >= 0.5)
        if sigFigs < 2: continue
        cleanFeatureNames = [c.replace(' ', '\n') for c in activeColumns]
        validShapPlots.append({
            'target': targetCol,
            'profile': profileName,
            'shapValues': shapValues,
            'displayX': displayX,
            'features': cleanFeatureNames,
            'algo': algoTag
        })
    

# Calculate grid size (Force 3 columns wide, variable rows)
numCols = 5
numRows = int(np.ceil(len(validShapPlots) / numCols))

# Allocate 8 inches per column and 6 inches per row
figShap, axesShap = plt.subplots(nrows=numRows, ncols=numCols, figsize=(24, 6 * numRows))
figShap.suptitle("Local Expert Predictors: Significant SHAP Impacts Only", fontsize=22, fontweight='bold')
axesShap = axesShap.flatten()

for idx, plotData in enumerate(validShapPlots):
    ax = axesShap[idx]
    plt.sca(ax)
    shap.summary_plot(
        plotData['shapValues'],
        plotData['displayX'],
        feature_names=plotData['features'],
        max_display=5,
        show=False,
        plot_size=None,
        color_bar=False
    )
    ax.tick_params(axis='y', labelsize=9)
    for label in ax.get_yticklabels():
        label.set_linespacing(0.8)
    ax.set_title(f"{plotData['profile']}\nTarget: {plotData['target']} {plotData['algo']}", fontsize=14, fontweight='bold')
    ax.set_xlabel("SHAP Value (Impact)", fontsize=10)

# Turn off any unused grid slots at the end
for extraIdx in range(len(validShapPlots), len(axesShap)):
    axesShap[extraIdx].set_axis_off()

figShap.subplots_adjust(top=0.93, bottom=0.03, left=0.05, right=0.95, hspace=0.7, wspace=0.5)
plt.show()

# =======================================================
# PARTIAL DEPENDENCE & ICE: GLOBAL RISK CURVES BY TARGET
# =======================================================

def inverseLogFormat(x, pos):
    val = np.sign(x) * np.expm1(np.abs(x))
    return f"{val:.3g}"

pdpFormatter = FuncFormatter(inverseLogFormat)

pdpTargets = [
    'Condition Number', 'isSvdFailed', 'Rank Collapse',
    'Positive Definite', 'isIrreducible', 'isCholesky'
]

# Set up the 4 Row by 3 Column grid
figPDP, axesPDP = plt.subplots(nrows=4, ncols=3, figsize=(24, 32))
figPDP.suptitle("Partial Dependence Risk Curves & 2D Interactions (Top Global Features)", fontsize=24, fontweight='bold')

for tIdx, targetCol in enumerate(pdpTargets):
    # Calculate grid positions (1D on top, 2D directly below it)
    colIdx = tIdx % 3
    rowOneD = (tIdx // 3) * 2
    rowTwoD = rowOneD + 1
    axOneD = axesPDP[rowOneD, colIdx]
    axTwoD = axesPDP[rowTwoD, colIdx]
    pdpData = dfTransformed[[targetCol] + expertFeatures].copy()

    if targetCol == 'Condition Number':
        pdpData[targetCol] = pd.to_numeric(pdpData[targetCol], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(1e35)
    else:
        if pdpData[targetCol].dtype == object:
            pdpData[targetCol] = pdpData[targetCol].replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0})
        pdpData[targetCol] = pd.to_numeric(pdpData[targetCol], errors='coerce')
    pdpData = pdpData.dropna(subset=[targetCol])

    if len(pdpData) < 20:
        axOneD.set_axis_off()
        axTwoD.set_axis_off()
        continue

    xPdp = pdpData[expertFeatures].copy().replace([np.inf, -np.inf], [1e35, -1e35])
    if 'Minimum Singular Value' in xPdp.columns:
        xPdp['Minimum Singular Value'] = xPdp['Minimum Singular Value'].fillna(0.0)
    for colName in expertFeatures:
        safeCol = pd.to_numeric(xPdp[colName], errors='coerce').astype(float)
        xPdp[colName] = np.sign(safeCol) * np.log1p(np.abs(safeCol))

    yPdp = pd.to_numeric(pdpData[targetCol], errors='coerce').astype(float)
    isClassification = targetCol in ['Rank Collapse', 'Positive Definite', 'isSvdFailed', 'isIrreducible', 'isCholesky']
    cleanX = xPdp.copy()
    # Check Global Dominance to Route PDP
    targetGaps = dfProof[dfProof['Target'] == targetCol]['Non-Linear Gap']
    globalIsLinear = True if (targetGaps.mean() < -0.02) else False
    # Create temporary imputed space for accurate K-NN distance in Mutual Information
    cleanXImputed = SimpleImputer(strategy='median', add_indicator=True).fit_transform(cleanX)

    if targetCol == 'Condition Number':
        yModel = np.log1p(np.clip(yPdp, 0, 1e35))
        miScores = mutual_info_regression(cleanXImputed, yModel, random_state=42)
    elif isClassification:
        yModel = (yPdp > 0).astype(int)
        if len(np.unique(yModel)) < 2:
            axOneD.set_axis_off()
            axTwoD.set_axis_off()
            continue
        miScores = mutual_info_classif(cleanXImputed, yModel, random_state=42)
    else:
        qt = QuantileTransformer(output_distribution='uniform', random_state=42)
        yModel = qt.fit_transform(yPdp.values.reshape(-1, 1)).ravel()
        miScores = mutual_info_regression(cleanXImputed, yModel, random_state=42)
    miSeries = pd.Series(miScores[:len(cleanX.columns)], index=cleanX.columns)
    if isClassification: targetCorrelation = miSeries
    else: targetCorrelation = cleanX.corrwith(pd.Series(yPdp, index=cleanX.index), method='spearman').abs()
    corrMatrix = cleanX.corr(method='spearman').abs()
    featuresToDrop = set()

    for i in range(len(corrMatrix.columns)):
        for j in range(i+1, len(corrMatrix.columns)):
            col1, col2 = corrMatrix.columns[i], corrMatrix.columns[j]
            if corrMatrix.iloc[i, j] > 0.85:
                if targetCorrelation[col1] > targetCorrelation[col2]: featuresToDrop.add(col2)
                else: featuresToDrop.add(col1)
    prunedFeatures = [f for f in cleanX.columns if f not in featuresToDrop]
    prunedMI = miSeries[prunedFeatures].sort_values(ascending=False)

    if len(prunedMI) < 2:
        axOneD.set_axis_off()
        axTwoD.set_axis_off()
        continue
    topFeature = prunedMI.index[0]
    topTwoFeatures = prunedMI.index[:2].tolist()
    displayX = cleanX[prunedFeatures]

    if globalIsLinear:
        if isClassification: pdpModel = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(penalty='l1', solver='saga', C=0.1, random_state=42, max_iter=5000, class_weight='balanced'))
        else: pdpModel = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), Lasso(alpha=0.05, random_state=42, max_iter=10000))
        algoLabel = "[L1 Linear]"
    else:
        if isClassification:
            posCount = sum(yModel)
            posWeight = (len(yModel) - posCount) / posCount if posCount > 0 else 1
            pdpModel = xgb.XGBClassifier(n_estimators=75, max_depth=4, random_state=42, scale_pos_weight=posWeight)
        else: pdpModel = xgb.XGBRegressor(n_estimators=75, max_depth=4, random_state=42)
        algoLabel = "[XGBoost]"
    pdpModel.fit(displayX, yModel)

    # ==========================
    # 1D PDP PLOT (Top Driver)
    # ==========================
    PartialDependenceDisplay.from_estimator(
        pdpModel, displayX, [topFeature], kind='both', ax=axOneD,
        ice_lines_kw={"color": "grey", "alpha": 0.1, "linewidth": 0.5},
        pd_line_kw={"color": "red", "linewidth": 3}
    )
    axOneD.xaxis.set_major_formatter(pdpFormatter)
    axOneD.set_title(f"Target: {targetCol} {algoLabel}\nDominant Driver: {topFeature}", fontsize=14, fontweight='bold')

    # ==========================
    # 2D INTERACTION PLOT
    # ==========================
    featOne, featTwo = topTwoFeatures
    try:
        displayTwoD = PartialDependenceDisplay.from_estimator(
            pdpModel, displayX, [(featOne, featTwo)], kind='average', ax=axTwoD,
            percentiles=(0.05, 0.95), contour_kw={'cmap': 'coolwarm', 'alpha': 0.8}
        )
        # Reverse Condition Number exponentiation to match real scale, keep others as prob/standard metric
        if targetCol == 'Condition Number' and hasattr(displayTwoD, 'contours_'):
            csList = displayTwoD.contours_.flatten()
            if len(csList) > 0:
                cs = csList[0]
                if hasattr(cs, 'labelTexts'):
                    for txt in cs.labelTexts: txt.set_visible(False)
                handles, labels = [], []
                for level in cs.levels:
                    color = cs.cmap(cs.norm(level))
                    handles.append(plt.Line2D([], [], color=color, lw=3))
                    origTargetVal = np.expm1(level)
                    labels.append(f"{origTargetVal:.3g}")
                axTwoD.legend(handles, labels, loc='upper right', title="Cond. Number", fontsize=9, title_fontproperties={'weight':'bold'}, framealpha=0.9)
        axTwoD.xaxis.set_major_formatter(pdpFormatter)
        axTwoD.yaxis.set_major_formatter(pdpFormatter)
        axTwoD.set_title(f"Interaction:\n{featOne} & {featTwo}", fontsize=13, fontweight='bold')

    except Exception as e:
        axTwoD.set_title(f"2D Interaction Render Failed:\n{str(e)}", fontsize=10, color='maroon')
        axTwoD.set_axis_off()

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.show()

# ====================================
# SHAP DEPENDENCE: ROOT CAUSE ANALYSIS
# ====================================

riskAnalysisTargets = ['Condition Number', 'isSvdFailed', 'Rank Collapse', 'Positive Definite', 'isIrreducible']

figRisk, axesRisk = plt.subplots(nrows=len(riskAnalysisTargets), ncols=2, figsize=(26, 8 * len(riskAnalysisTargets)))
figRisk.suptitle("Root Cause Analysis: Unifying Decision Trees, XGBoost, and Lasso Regression", fontsize=24, fontweight='bold')

def wrap_text(textData):
    if pd.isna(textData): return "Unknown"
    cleanText = str(textData).replace('\n', ' ')
    wordList = cleanText.split()
    return "\n".join([" ".join(wordList[i:i+2]) for i in range(0, len(wordList), 2)])

for tIdx, targetCol in enumerate(riskAnalysisTargets):
    axRules = axesRisk[tIdx, 0]
    axRules.set_axis_off()
    axShap = axesRisk[tIdx, 1]
    treeData = dfTransformed[treeFeatures + [targetCol]].copy()
    if targetCol == 'Rank Collapse':
        treeData[targetCol] = pd.to_numeric(treeData[targetCol], errors='coerce').replace([np.inf, -np.inf], np.nan)
    else: treeData[targetCol] = pd.to_numeric(treeData[targetCol], errors='coerce')
    treeData = treeData.dropna(subset=[targetCol])
    for col in treeFeatures:
        if col != 'isSquare':
            treeData[col] = pd.to_numeric(treeData[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(-1e5)
    if targetCol == 'Condition Number':
        yTreeBinary = (treeData[targetCol] > 1e4).astype(int)
    else:
        yTreeBinary = (treeData[targetCol] > 0).astype(int)
    treeData['RiskFlag'] = yTreeBinary

    @cacheDir.cache
    def train_rca_subtree(dataSubset, matrixShape):
        if len(dataSubset) == 0 or len(np.unique(dataSubset['RiskFlag'])) < 2:
            prob = dataSubset['RiskFlag'].mean() if len(dataSubset) > 0 else 0.0
            return {'type': 'leaf', 'class': f"Risk: {prob:.0%}", 'samples': len(dataSubset), 'risk_prob': prob}
        xSub = dataSubset[[c for c in treeFeatures if c != 'isSquare']]
        ySub = dataSubset['RiskFlag']
        # Monte-Carlo GridSearchCV with CCP Pruning for the subtree
        paramGrid = {'max_depth': [2, 3, 4], 'min_samples_leaf': [5, 10], 'ccp_alpha': [0.0, 0.015]}
        cvSplits = max(3, min(5, sum(ySub), len(ySub) - sum(ySub)))
        mcCv = StratifiedShuffleSplit(n_splits=cvSplits, test_size=0.2, random_state=42)
        baseTree = DecisionTreeClassifier(random_state=42)
        grid = GridSearchCV(baseTree, paramGrid, cv=mcCv, scoring='balanced_accuracy')
        grid.fit(xSub, ySub)
        bestModel = grid.best_estimator_
        tData = bestModel.tree_
        subFeatures = xSub.columns

        def extract_node(nodeId):
            if tData.children_left[nodeId] == -1:
                vals = tData.value[nodeId][0]
                prob = vals[1] / np.sum(vals) if len(vals) > 1 else 0.0
                return {'type': 'leaf', 'class': f"Risk:\n{prob:.0%}", 'samples': tData.n_node_samples[nodeId], 'risk_prob': prob}
            return {
                'type': 'internal', 'feature': subFeatures[tData.feature[nodeId]], 'threshold': tData.threshold[nodeId],
                'samples': tData.n_node_samples[nodeId], 'left': extract_node(tData.children_left[nodeId]), 'right': extract_node(tData.children_right[nodeId])
            }
        return extract_node(0)
    sqData = treeData[treeData['isSquare'] == 1].copy()
    rectData = treeData[treeData['isSquare'] == 0].copy()
    unifiedRcaTree = {
        'type': 'internal', 'feature': 'isSquare', 'threshold': 0.5, 'samples': len(treeData),
        'left': train_rca_subtree(rectData, 'Rect'), 'right': train_rca_subtree(sqData, 'Square')
    }

    # Render the Custom Unified Tree
    def set_rca_coords(nodeDict, depthLvl, xOffset):
        if nodeDict is None: return xOffset
        if nodeDict['type'] == 'leaf':
            nodeDict['x'], nodeDict['y'] = xOffset, -depthLvl
            return xOffset + 1.5
        leftOffset = set_rca_coords(nodeDict['left'], depthLvl + 1, xOffset)
        rightOffset = set_rca_coords(nodeDict['right'], depthLvl + 1, leftOffset)
        nodeDict['x'], nodeDict['y'] = (nodeDict['left']['x'] + nodeDict['right']['x']) / 2.0, -depthLvl
        return rightOffset
    set_rca_coords(unifiedRcaTree, 0, 0.0)

    def draw_rca_tree(nodeDict, ax):
        if nodeDict is None: return
        if nodeDict['type'] == 'leaf':
            nodeText = f"{nodeDict['class']}\n(n={nodeDict['samples']})"
            boxColor = '#fca5a5' if nodeDict['risk_prob'] > 0.5 else '#bbf7d0'
        else:
            if nodeDict['threshold'] < -10000: conditionText = "Computation Failed / Missing"
            else: conditionText = f"<= {nodeDict['threshold']:.2f}"
            nodeText = f"{wrap_text(nodeDict['feature'])}\n{conditionText}\n(n={nodeDict['samples']})"
            boxColor = '#fef08a'
        ax.text(nodeDict['x'], nodeDict['y'], nodeText, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor=boxColor, edgecolor="black", lw=1), fontsize=8, zorder=3)
        if nodeDict['type'] == 'internal':
            lX, lY, rX, rY = nodeDict['left']['x'], nodeDict['left']['y'], nodeDict['right']['x'], nodeDict['right']['y']
            # Draw connecting lines
            ax.plot([nodeDict['x'], lX], [nodeDict['y'], lY], 'k-', zorder=1)
            ax.plot([nodeDict['x'], rX], [nodeDict['y'], rY], 'k-', zorder=1)
            # Explicit True/False labels mapped to the branch midpoints
            ax.text((nodeDict['x'] + lX) / 2, (nodeDict['y'] + lY) / 2, r'$\leq$ True',
                    ha='center', va='center', fontsize=8, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.8), zorder=2)
            ax.text((nodeDict['x'] + rX) / 2, (nodeDict['y'] + rY) / 2, r'$>$ False',
                    ha='center', va='center', fontsize=8, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.8), zorder=2)
            draw_rca_tree(nodeDict['left'], ax)
            draw_rca_tree(nodeDict['right'], ax)
    draw_rca_tree(unifiedRcaTree, axRules)
    allX, allY = [], []

    def get_bnds(n):
        if n is None: return
        allX.append(n['x']); allY.append(n['y'])
        if n['type'] == 'internal': get_bnds(n['left']); get_bnds(n['right'])
    get_bnds(unifiedRcaTree)
    if allX and allY:
        axRules.set_xlim(min(allX) - 1, max(allX) + 1)
        axRules.set_ylim(min(allY) - 0.5, max(allY) + 0.5)
    axRules.set_title(f"Gating Tree: Isolating High-Risk Manifolds for '{targetCol}'", fontsize=15, fontweight='bold')

    def predict_rca_tree(row, node):
        if node['type'] == 'leaf': return node['risk_prob']
        val = row.get(node['feature'], -1e5)
        if pd.isna(val): val = -1e5
        return predict_rca_tree(row, node['left']) if val <= node['threshold'] else predict_rca_tree(row, node['right'])

    # Extract the High Risk Regime from the Custom Tree
    treeData['TreeRiskProb'] = treeData.apply(lambda r: predict_rca_tree(r, unifiedRcaTree), axis=1)
    highRiskRegimeDf = treeData[treeData['TreeRiskProb'] > 0.5].copy()
    highRiskRegimeDf = dfTransformed.loc[highRiskRegimeDf.index, [targetCol] + expertFeatures].copy()
    if len(highRiskRegimeDf) < 20:
        axShap.set_axis_off()
        axShap.text(0.5, 0.5, f"Insufficient matrices in High-Risk Regime\nto train local expert for {targetCol}", ha='center', va='center', fontsize=12, color='maroon')
        continue
    # Setup the Local Explainer Target
    if targetCol == 'Condition Number':
        yRiskRaw = highRiskRegimeDf[targetCol].replace([np.inf, -np.inf], np.nan).fillna(1e35)
        yRisk = np.log1p(np.clip(yRiskRaw, 0, 1e35)) # Log scale for continuous Lasso
        isLinear = True
    else:
        yRisk = (highRiskRegimeDf[targetCol] > 0).astype(int)
        isLinear = False
    cleanRiskX = highRiskRegimeDf[expertFeatures].copy()
    for colName in expertFeatures:
        if colName in ['Minimum Singular Value', 'Density', 'Diagonally Dominant\nRow Fraction', 'RCM Compression Ratio']:
            cleanRiskX[colName] = cleanRiskX[colName].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            cleanRiskX[colName] = cleanRiskX[colName].replace([np.inf, -np.inf], np.nan).fillna(1e35)
        safeCol = pd.to_numeric(cleanRiskX[colName], errors='coerce').astype(float)
        cleanRiskX[colName] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
    if isLinear: targetCorr = cleanRiskX.corrwith(pd.Series(yRisk, index=cleanRiskX.index), method='spearman').abs()
    else: targetCorr = cleanRiskX.corrwith(pd.Series(yRisk, index=cleanRiskX.index), method='spearman').abs()
    corrMatrixRisk = cleanRiskX.corr(method='spearman').abs()
    dropRisk = set()
    for i in range(len(corrMatrixRisk.columns)):
        for j in range(i+1, len(corrMatrixRisk.columns)):
            col1, col2 = corrMatrixRisk.columns[i], corrMatrixRisk.columns[j]
            if corrMatrixRisk.iloc[i, j] > 0.85:
                if targetCorr[col1] > targetCorr[col2]: dropRisk.add(col2)
                else: dropRisk.add(col1)
    prunedRiskFeatures = [f for f in expertFeatures if f not in dropRisk]
    xRiskUnscaled = cleanRiskX[prunedRiskFeatures]
    if not isLinear:
        posCount = sum(yRisk)
        negCount = len(yRisk) - posCount
        if posCount < 3 or negCount < 3:
            axShap.set_axis_off()
            axShap.text(0.5, 0.5, f"Target: {targetCol}\n\n[ Axiomatic State ]\nVariance too low for SHAP.\n(Target constraint is inherent to this topological cluster)",
                        ha='center', va='center', fontsize=13, color='maroon', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='maroon', lw=2))
            continue
        posWeight = negCount / posCount if posCount > 0 else 1
        expertModel = xgb.XGBClassifier(n_estimators=75, max_depth=4, learning_rate=0.1, random_state=42, scale_pos_weight=posWeight)
        expertModel.fit(xRiskUnscaled, yRisk)
        explainer = shap.TreeExplainer(expertModel)
        shapValues = explainer.shap_values(xRiskUnscaled)
    else:
        xRiskScaled = pd.DataFrame(StandardScaler().fit_transform(xRiskUnscaled), columns=xRiskUnscaled.columns, index=xRiskUnscaled.index)
        expertModel = Lasso(alpha=0.05, random_state=42, max_iter=10000)
        expertModel.fit(xRiskScaled, yRisk)
        explainer = shap.LinearExplainer(expertModel, xRiskScaled)
        shapValues = explainer.shap_values(xRiskScaled)
    # Extract SHAP values
    if isinstance(shapValues, list): shapValues = shapValues[1] if len(shapValues > 1) else shapValues[0]
    if len(shapValues.shape) == 3: shapValues = shapValues[:, :, 1]
    localImportances = pd.Series(np.abs(shapValues).mean(axis=0), index=prunedRiskFeatures).sort_values(ascending=False)
    topExpertFeature = localImportances.index[0]
    topExpertFeatureIndex = prunedRiskFeatures.index(topExpertFeature)

    plt.sca(axShap)
    try:
        xData = xRiskUnscaled.iloc[:, topExpertFeatureIndex]
        yShap = shapValues[:, topExpertFeatureIndex] if len(shapValues.shape) > 1 else shapValues
        axKde = axShap.twinx()
        sns.kdeplot(x=xData, ax=axKde, color='slategrey', fill=True, alpha=0.2, linewidth=1.5, zorder=1)
        axKde.set_ylim(bottom=0)
        axKde.set_yticks([])
        axKde.set_ylabel("")
        scatter = axShap.scatter(xData, yShap, c=xData, cmap='coolwarm', alpha=0.8, s=90, edgecolor='black', linewidth=0.5, zorder=3)
        axShap.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Neutral Risk Baseline (SHAP = 0)', zorder=2)
        plt.colorbar(scatter, ax=axShap, pad=0.02, label='Native Log Magnitude')
        # Force the scatter axis to render on top, but make its background transparent so the KDE shows through
        axShap.set_zorder(axKde.get_zorder() + 1)
        axShap.patch.set_visible(False)
        algoLabel = "Lasso Regression" if isLinear else "XGBoost"
        axShap.set_title(f"Local Explainer ({algoLabel}): SHAP Impact of Top Feature\n{topExpertFeature}", fontsize=15, fontweight='bold')
        axShap.set_ylabel("SHAP Value (Risk Impact)", fontsize=12)
        axShap.set_xlabel(f"Magnitude: {topExpertFeature}", fontsize=12, fontweight='bold')
        axShap.legend(loc='upper left', fontsize=10)
        axShap.grid(True, linestyle=':', alpha=0.5)
    except:
        axShap.set_title(f"SHAP Dependence Render Failed for {topExpertFeature}", fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# =========================================================
# EMPIRICAL TIMING BENCHMARK (Local Matrix Parsing)
# =========================================================

if os.path.exists('./cached_computations/timing_benchmark_checkpoint.joblib'): dfTiming = joblib.load('./cached_computations/timing_benchmark_checkpoint.joblib')
else: 
    timingRecords = []
    baseDir = Path('.')
    matrixDir = baseDir / 'Matrix Files'
    csvPath = Path('matrixdata.csv')
    validMatrixNames = set()
    if csvPath.exists():
        dfMeta = pd.read_csv(csvPath)
        dfValid = dfMeta.dropna(subset=['Minimum Singular Value'])
        validMatrixNames = set(dfValid['Name'].astype(str).str.strip())
    
    if matrixDir.exists() and os.path.isdir(matrixDir):
        mtxFiles = [f for f in os.listdir(matrixDir) if f.endswith('.mtx')]
        tempData = []
        for fileName in mtxFiles:
            matName = fileName.replace('.mtx', '').strip()
            if validMatrixNames and matName not in validMatrixNames:
                continue
            try:
                matPath = os.path.join(matrixDir, fileName)
                matProxy = mmread(matPath)
                matSize = matProxy.shape[0] * matProxy.shape[1]
                matDensity = matProxy.nnz / matSize if matSize > 0 else 0
                tempData.append({'file': fileName, 'size': matSize, 'density': matDensity, 'path': matPath})
            except: pass
        if tempData:
            dfTemp = pd.DataFrame(tempData)
            if len(dfTemp) > 1:
                meanSize = dfTemp['size'].mean()
                stdSize = dfTemp['size'].std()
                dfFiltered = dfTemp[(dfTemp['size'] >= meanSize - stdSize) & (dfTemp['size'] <= meanSize + stdSize)]
            else:
                dfFiltered = dfTemp
            for _, rowObj in dfFiltered.iterrows():
                try:
                    aMat = mmread(rowObj['path']).tocsr()
                    nSize = aMat.shape[0]
                    featureTimes = {}
                    # t0: isSquare (Stage 0)
                    t0 = time.perf_counter()
                    isSquareVal = aMat.shape[0] == aMat.shape[1]
                    featureTimes['isSquare\nMatrix Archetype'] = (time.perf_counter() - t0) * 1000
                    aAlg = sp.bmat([[None, aMat], [aMat.T, None]], format='csr')
                    aTop = laplacian(sp.bmat([[None, np.abs(aMat)], [np.abs(aMat).T, None]], format='csr')).tocsr()
                    # t1: Diagonally Dominant Row Fraction
                    t1 = time.perf_counter()
                    aAlg = sp.bmat([[None, aMat], [aMat.T, None]], format='csr')
                    rowSumsAlg = np.array(np.abs(aAlg).sum(axis=1)).flatten()
                    time_t1 = (time.perf_counter() - t1) * 1000
                    featureTimes['Diagonally Dominant\nRow Fraction'] = time_t1
                    # t2: Directional Mean Bias & Frobenius
                    t2 = time.perf_counter()
                    aAlg = sp.bmat([[None, aMat], [aMat.T, None]], format='csr')
                    dataArr = np.nan_to_num(aAlg.data, nan=0.0, posinf=1e15, neginf=-1e15)
                    posData = dataArr[dataArr > 0]
                    negData = dataArr[dataArr < 0]
                    meanP = np.mean(posData) if len(posData) > 0 else 0.0
                    meanN = np.mean(negData) if len(negData) > 0 else 0.0
                    time_t2 = (time.perf_counter() - t2) * 1000
                    featureTimes['Directional Mean Bias'] = time_t2
                    featureTimes['Signed Frobenius Ratio'] = time_t2
                    # t3: Brauer Max Product & Degeneracy
                    t3 = time.perf_counter()
                    aAlg = sp.bmat([[None, aMat], [aMat.T, None]], format='csr')
                    rows, cols = aAlg.nonzero()
                    if len(rows) > 0:
                        products = rowSumsAlg[rows] * rowSumsAlg[cols]
                    time_t3 = (time.perf_counter() - t3) * 1000
                    featureTimes['Brauer Max Product'] = time_t3
                    featureTimes['Degeneracy Multiplier'] = time_t3
                    # t4: Topological Entropy (Explicitly defined as sum of t2 and t3)
                    time_t4 = time_t2 + time_t3
                    featureTimes['Topological Entropy'] = time_t4
                    # t5: RCM Bandwidth
                    t5 = time.perf_counter()
                    aTop = laplacian(sp.bmat([[None, np.abs(aMat)], [np.abs(aMat).T, None]], format='csr')).tocsr()
                    try: _ = reverse_cuthill_mckee(aTop, symmetric_mode=True)
                    except: pass
                    time_t5 = (time.perf_counter() - t5) * 1000
                    featureTimes['RCM Bandwidth'] = time_t5
                    # t6: Fiedler Value
                    try:
                        t6 = time.perf_counter()
                        aTop = laplacian(sp.bmat([[None, np.abs(aMat)], [np.abs(aMat).T, None]], format='csr')).tocsr()
                        aSym = 0.5 * (aTop + aTop.T).astype(np.float64)
                        if aSym.shape[0] > 2:
                            _, _ = splinalg.eigsh(aSym, k=2, which='SA', maxiter=1000)
                        time_t6 = (time.perf_counter() - t6) * 1000
                        featureTimes['Fiedler Value'] = time_t6
                    except: pass
                    timingRecords.append({'Density': rowObj['density'], **featureTimes})
                except Exception as e:
                    print(f"Failed: {e}")
    
    dfTiming = pd.DataFrame(timingRecords)
    joblib.dump(dfTiming, './cached_computations/timing_benchmark_checkpoint.joblib')

# --------------------------------
# DEFINE PROGRESSIVE ROUTING GATES
# --------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)

stage1Features = [
    'isSquare', 'Diagonally Dominant\nRow Fraction', 'Directional Mean Bias', 'Density', 'Brauer Ratio',
    'Brauer Max Product', 'Signed Frobenius Ratio', 'Degeneracy Multiplier', 'Brauer Min Product',
    'Brauer Mean Product (Top)', 'Brauer Mean Center Distance', 'Brauer Max Center Distance'
]

stage2Features = [
    'isSquare', 'RCM Bandwidth', 'Topological Entropy', 'RCM Compression Ratio'
]

stage3Features = ['Fiedler Value']

if 'isDecomposable' not in dfTransformed.columns:
    dfTransformed['isDecomposable'] = (pd.to_numeric(dfTransformed['Num Dmperm Blocks'], errors='coerce').fillna(0) >= 5).astype(int)

if 'sprank(A)-rank(A)' in dfTransformed.columns:
    degCol = pd.to_numeric(dfTransformed['sprank(A)-rank(A)'], errors='coerce')
    dfTransformed['isDegenerate'] = np.where(degCol.isna(), np.nan, (degCol > 0).astype(int))

routingTargets = {
    'Rank Collapse': {'features': stage1Features, 'action': 'Route to Rank-Revealing QR (RRQR) / TSVD'},
    'isSvdFailed': {'features': stage1Features, 'action': 'Reject / Route to Arbitrary Precision Arithmetic'},
    'Positive Definite': {'features': stage1Features, 'action': 'Route to Fast Cholesky Factorization'},
    'isCholesky': {'features': stage1Features, 'action': 'Route to Fast Cholesky Factorization'},
    'isIrreducible': {
        'features': list(set(stage1Features + stage2Features + stage3Features)),
        'action': 'Route to Multifrontal LU (MUMPS) / LDLT'
    },
    'isDecomposable': {
        'features': list(set(stage1Features + stage2Features + stage3Features)),
        'action': 'Route to Dantzig-Wolfe / Parallel ADMM'
    },
    'isDegenerate': {
        'features': list(set(stage1Features + stage2Features + stage3Features)),
        'action': 'Route to Exterior Point / Dual Simplex'
    }
}

if os.path.exists('./cached_computations/routing_models_checkpoint.joblib'):
    print("CONSOLE LOG: Loading pre-trained progressive routing models...")
    checkpoint = joblib.load('./cached_computations/routing_models_checkpoint.joblib')
    learnedThresholdsSquare = checkpoint['learnedThresholdsSquare']
    learnedThresholdsRect = checkpoint['learnedThresholdsRect']
    accuracyMetricsSquare = checkpoint['accuracyMetricsSquare']
    accuracyMetricsRect = checkpoint['accuracyMetricsRect']
else:
    learnedThresholdsSquare = {}
    learnedThresholdsRect = {}
    accuracyMetricsSquare = []
    accuracyMetricsRect = []
    
    shapeSplits = { 'Square': dfTransformed[dfTransformed['isSquare'] == 1].copy(),
                   'Rect': dfTransformed[dfTransformed['isSquare'] == 0].copy()}

    for shapeName, shapeData in shapeSplits.items():
        for targetName, configDetails in routingTargets.items():
            if shapeName == 'Rect' and targetName in ['isCholesky', 'isIrreducible', 'Positive Definite']: continue
            allowedFeatures = configDetails['features']
            
            if shapeName == 'Rect': 
                squareOnlyFeatures = ['Skew-Symmetric Frobenius Norm', 'Diagonal Dominance Min Ratio']
                allowedFeatures = [f for f in allowedFeatures if f not in squareOnlyFeatures]
            treeData = shapeData[allowedFeatures + [targetName]].copy()
            if treeData[targetName].dtype == object:
                treeData[targetName] = treeData[targetName].replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0})
            if targetName == 'Rank Collapse': treeData[targetName] = pd.to_numeric(treeData[targetName], errors='coerce').fillna(1)
            else: treeData[targetName] = pd.to_numeric(treeData[targetName], errors='coerce')
            treeData = treeData.dropna(subset=[targetName])
            for col in allowedFeatures:
                safeCol = pd.to_numeric(treeData[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
                treeData[col] = np.clip(safeCol, a_min=-1e35, a_max=1e35)
            xTree = treeData[allowedFeatures]
            yTree = (treeData[targetName] > 0).astype(int)
            
            posCount = sum(yTree)
            negCount = len(yTree) - posCount
            if posCount >= 3 and negCount >= 3:
                # 1. Base Stratified Split (xTrain for CV Optimization, xVal reserved strictly for Calibration)
                xTrainRaw, xValRaw, yTrain, yVal = train_test_split(xTree, yTree, test_size=0.25, stratify=yTree, random_state=42)
                
                targetCorr = xTrainRaw.corrwith(pd.Series(yTree, index=xTrainRaw.index), method='spearman').abs()
                corrMatrix = xTrainRaw.corr(method='spearman').abs()
                featuresToDrop = set()
                for i in range(len(corrMatrix.columns)):
                    for j in range(i+1, len(corrMatrix.columns)):
                        colA, colB = corrMatrix.columns[i], corrMatrix.columns[j]
                        if corrMatrix.iloc[i, j] > 0.85:
                            if targetCorr[colA] > targetCorr[colB]: featuresToDrop.add(colB)
                            else: featuresToDrop.add(colA)
                prunedFeatures = [f for f in allowedFeatures if f not in featuresToDrop]
                xTrain = xTrainRaw[prunedFeatures]
                xVal = xValRaw[prunedFeatures]
                
                # 2. Robust Bayesian Hyperparameter Sweep (Optuna with Internal CV)
                def objective(trial):
                    suggestedDepth = trial.suggest_int('max_depth', 2, 6)
                    growPolicy = trial.suggest_categorical('grow_policy', {'depthwise', 'lossguide'})
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': suggestedDepth,
                        'grow_policy': growPolicy,
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                        'subsample': trial.suggest_float('subsample', 0.8, 1.0), # Aggressive row sampling
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9), # Aggressive feature sampling
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 0.5, 5), # Prevent outlier isolation
                        'max_delta_step': trial.suggest_int('max_delta_step', 1, 7),
                        'gamma': trial.suggest_float('gamma', 0.01, 3.0, log=True), # Strict loss-reduction threshold
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 100.0, log=True),
                        'scale_pos_weight': 1.0,
                        'n_jobs': -1
                    }
    
                    if growPolicy == 'lossguide': params['max_leaves'] = trial.suggest_int('max_leaves' , 10, 31)
                    # Internal Stratified K-Fold to prevent validation-set overfitting
                    maxSplits = min(5, sum(yTrain), len(yTrain) - sum(yTrain))
                    cvFold = StratifiedKFold(n_splits=maxSplits, shuffle=True, random_state=42)
                    foldScores = []
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Evaluate params across 3 different cuts of the training data
                        for trainIdx, testIdx in cvFold.split(xTrain, yTrain):
                            xFoldTrain, xFoldTest = xTrain.iloc[trainIdx], xTrain.iloc[testIdx]
                            yFoldTrain, yFoldTest = yTrain.iloc[trainIdx], yTrain.iloc[testIdx]
                            model = xgb.XGBClassifier(**params, random_state=42, eval_metric='aucpr')
                            model.fit(xFoldTrain, yFoldTrain, verbose=False)
                            yFoldProb = model.predict_proba(xFoldTest)[:, 1]
                            foldScores.append(average_precision_score(yFoldTest, yFoldProb))
                    return np.mean(foldScores)
    
                study = optuna.create_study(direction='maximize')
                nTrials = 200 if shapeName == 'Square' else 75
                study.optimize(objective, n_trials=nTrials)
                # 3. Train best raw model
                bestXgb = xgb.XGBClassifier(**study.best_params, random_state=42, eval_metric='aucpr', early_stopping_rounds=15)
                bestXgb.fit(xTrain, yTrain, eval_set=[(xVal, yVal)], verbose=False)
                optimalTrees = (bestXgb.best_iteration + 1) if hasattr(bestXgb, 'best_iteration') else study.best_params.get('n_estimators', 100)
                # 4. Isotonic Probability Calibration (Dynamic Folding)
                safeParam = study.best_params.copy()
                if 'n_estimators' in safeParam: del safeParam['n_estimators']
                calibrateBase = xgb.XGBClassifier(**safeParam, n_estimators=optimalTrees, random_state=42, eval_metric='aucpr')
                trainPosCt = sum(yTrain)
                trainNegCt = len(yTrain) - trainPosCt
                calibCv = int(min(5, trainPosCt, trainNegCt))
                calibMethod = 'sigmoid' if min(posCount, negCount) < 15 else 'isotonic'
                if calibCv < 2:
                    calibrateBase.fit(xTrain, yTrain)
                    calibratedModel = CalibratedClassifierCV(estimator=calibrateBase, method=calibMethod, cv='prefit')
                    calibratedModel.fit(xVal, yVal)
                else:
                    calibratedModel = CalibratedClassifierCV(estimator=calibrateBase, method=calibMethod, cv=calibCv)
                    calibratedModel.fit(xTrain, yTrain)
                # 5. Dynamic Threshold Extraction
                yTrainProbs = calibratedModel.predict_proba(xTrain)[:, 1]
                yValProbs = calibratedModel.predict_proba(xVal)[:, 1]
                bestMcc = -1.0
                optimalThreshold = 0.5
                # Granular sweep to find the exact inflection point of maximum quadrant balance
                thresholdCandidates = np.linspace(0.05, 0.95, 100)
                for thresh in thresholdCandidates:
                    yTrainPredSweep = (yTrainProbs >= thresh).astype(int)
                    if len(np.unique(yTrainPredSweep)) == 1: continue
                    currentMcc = matthews_corrcoef(yTrain, yTrainPredSweep)
                    if currentMcc > bestMcc:
                        bestMcc = currentMcc
                        optimalThreshold = thresh
                # Fallback to empirical ratio if MCC optimization fails to find a valid split
                if bestMcc == -1.0: optimalThreshold = max(0.1, np.mean(yVal))
                # 6. Evaluate finalized dynamic model using the MCC-optimized threshold
                yValPred = (yValProbs >= optimalThreshold).astype(int)
                balancedAccVal = balanced_accuracy_score(yVal, yValPred)
                mccVal = matthews_corrcoef(yVal, yValPred)
                f1Val = f1_score(yVal, yValPred, zero_division=0)
                nNeighbors = min(3, len(xTree) - 1)
                xTreeMI = xTree.replace(-1e5, np.nan)
                xTreeImp = SimpleImputer(strategy='median', add_indicator=True).fit_transform(xTreeMI)
                miScores = mutual_info_classif(xTreeImp, yTree, n_neighbors=nNeighbors, random_state=42)
                miSeries = pd.Series(miScores[:len(prunedFeatures)], index=prunedFeatures).sort_values(ascending=False)
                topFeatureName = miSeries.index[0]
                meanConf = np.mean(yValProbs[yValPred == 1]) * 100 if sum(yValPred) > 0 else 0.0
    
                if shapeName == 'Square':
                    learnedThresholdsSquare[targetName] = {
                        'model': calibratedModel,
                        'features': prunedFeatures,
                        'action': configDetails['action'],
                        'invertLogic': configDetails.get('invertLogic', False),
                        'optimalThreshold': optimalThreshold,
                        'rootFeature': topFeatureName,
                        'meanConfidence': meanConf
                    }
                    accuracyMetricsSquare.append({
                        'Target Constraint': targetName,
                        'Gating Feature': topFeatureName,
                        'Balanced Acc': balancedAccVal,
                        'MCC': mccVal,
                        'F1 Score': f1Val
                    })
                else:
                    learnedThresholdsRect[targetName] = {
                        'model': calibratedModel,
                        'features': prunedFeatures,
                        'action': configDetails['action'],
                        'invertLogic': configDetails.get('invertLogic', False),
                        'optimalThreshold': optimalThreshold,
                        'rootFeature': topFeatureName,
                        'meanConfidence': meanConf
                    }
                    accuracyMetricsRect.append({
                        'Target Constraint': targetName,
                        'Gating Feature': topFeatureName,
                        'Balanced Acc': balancedAccVal,
                        'MCC': mccVal,
                        'F1 Score': f1Val
                    })

    joblib.dump({'learnedThresholdsRect': learnedThresholdsRect, 'learnedThresholdsSquare': learnedThresholdsSquare, 'accuracyMetricsSquare': accuracyMetricsSquare, 'accuracyMetricsRect': accuracyMetricsRect}, './cached_computations/routing_models_checkpoint.joblib')


# ---------------------------------------------------------
# EXECUTE AGENTIC ROUTER (COMBINATORIAL STATE MAP)
# ---------------------------------------------------------
solverAbbrevDict = {  
    'Route to Regularized Iterative (GMRES / LSQR)': 'GMRES / LSQR',
    'Route to Rank-Revealing QR (RRQR) / TSVD': 'RRQR / TSVD',
    'Route to Arbitrary Precision LU (MPFR)': 'AP: Dense LU',
    'Route to Arbitrary Precision GMRES (MPFR)': 'AP: Iterative GMRES',
    'Route to Multifrontal LDLT (MUMPS)': 'MUMPS (LDLT)',
    'Route to Centralized Direct LU / MUMPS': 'MUMPS (LU)',
    'Route to Fast Cholesky Factorization': 'Fast Cholesky',
    'Route to Dantzig-Wolfe / Parallel ADMM': 'Dantzig-Wolfe / ADMM',
    'Route to Exterior Point / Dual Simplex': 'Dual Simplex',
    'Route to Primal-Dual Interior Point Method (IPM)': 'Primal-Dual IPM'
}

def evaluate_deep_rule(row, targetKey):
    if row.get('isSquare', 1) == 1:
        rule = learnedThresholdsSquare.get(targetKey)
    else: rule = learnedThresholdsRect.get(targetKey)
    if not rule: return False, 0.0
    xVal = []
    for col in rule['features']:
        rawVal = pd.to_numeric(row.get(col, np.nan), errors='coerce')
        val = np.nan if pd.isna(rawVal) or np.isinf(rawVal) else rawVal
        xVal.append(np.clip(val, a_min=-1e35, a_max=1e35))
    xDf = pd.DataFrame([xVal], columns=rule['features'])
    prob = rule['model'].predict_proba(xDf)[0][1]
    safeThreshold = rule['optimalThreshold']
    pred = 1 if prob >= safeThreshold else 0
    if rule.get('invertLogic', False):
        return pred == 0, 1.0 - prob if pred == 0 else prob
    return pred == 1, prob

def agenticSolverRouter(row):
    isSquare = row.get('isSquare', 1) == 1
    
    # 1. Ultra-stable matrices (Bypass everything)
    predPD, _ = evaluate_deep_rule(row, 'Positive Definite')
    predChol, _ = evaluate_deep_rule(row, 'isCholesky')
    if predPD: predChol = True
    if predChol: return 'Route to Fast Cholesky Factorization', 1
    
    # 2. Singular/Rank-Deficient (Must precede SVD check)
    predRank, _ = evaluate_deep_rule(row, 'Rank Collapse')
    if predRank: return 'Route to Rank-Revealing QR (RRQR) / TSVD', 1
    
    # Non-Square iterative fallback
    if not isSquare:
        return 'Route to Regularized Iterative (GMRES / LSQR)', 2

    # 3. Ill-Conditioned but Full-Rank (AP Net)
    predSvd, _ = evaluate_deep_rule(row, 'isSvdFailed')
    if predSvd:
        return 'Route to Arbitrary Precision Arithmetic', 1
        
    # 4. Degenerate bases (Must precede MUMPS)
    predDegen, _ = evaluate_deep_rule(row, 'isDegenerate')
    if predDegen: return 'Route to Exterior Point / Dual Simplex', 2

    # 5. Highly Connected / Irreducible (MUMPS Split)
    predIrred, _ = evaluate_deep_rule(row, 'isIrreducible')
    if predIrred: 
        baseSymmetry = pd.to_numeric(row.get('Skew-Symmetric Frobenius Norm', 1.0), errors='coerce')
        if baseSymmetry < 1e-10:
            return 'Route to Multifrontal LDLT (MUMPS)', 2
        else:
            return 'Route to Centralized Direct LU / MUMPS', 2    
    
    predDecomp, _ = evaluate_deep_rule(row, 'isDecomposable')
    if predDecomp: return 'Route to Dantzig-Wolfe / Parallel ADMM', 2
    
    return 'Route to Primal-Dual Interior Point Method (IPM)', 3

def getGroundTruthSolver(row):
    isSquare = row.get('isSquare', 1) == 1
    
    # 1. Cholesky
    isChol = row.get('isCholesky', 0) == 1
    if pd.to_numeric(row.get('Positive Definite', 0), errors='coerce') == 1:
        isChol = True
    if isChol: return 'Route to Fast Cholesky Factorization'

    # 2. Rank Collapse
    rankCol = row.get('Rank Collapse', 0) == 1
    if rankCol: return 'Route to Rank-Revealing QR (RRQR) / TSVD'
    
    # Non-square fallback
    if not isSquare:
        return 'Route to Regularized Iterative (GMRES / LSQR)'

    # 3. SVD Failures
    svdFail = row.get('isSvdFailed', 0) == 1
    if svdFail:
        return 'Route to Arbitrary Precision Arithmetic'
        
    # 4. Degenerate
    degen = row.get('isDegenerate', 0) == 1
    if degen: return 'Route to Exterior Point / Dual Simplex'
    
    # 5. MUMPS
    irred = row.get('isIrreducible', 0) == 1
    if irred: 
        baseSymmetry = pd.to_numeric(row.get('Skew-Symmetric Frobenius Norm', 1.0), errors='coerce')
        if baseSymmetry < 1e-10:
            return 'Route to Multifrontal LDLT (MUMPS)'
        else:
            return 'Route to Centralized Direct LU / MUMPS'
    
    decomp = row.get('isDecomposable', 0) == 1
    if decomp: return 'Route to Dantzig-Wolfe / Parallel ADMM'
    
    return 'Route to Primal-Dual Interior Point Method (IPM)'

# 1. Generate the initial predictions using your XGBoost logic
routerOutputs = evalData.apply(agenticSolverRouter, axis=1)
evalData['assignedSolver'] = [res[0] for res in routerOutputs]
evalData['computationStageRequired'] = [res[1] for res in routerOutputs]
dfSquare = evalData[evalData['isSquare'] == 1].copy()
dfRectangular = evalData[evalData['isSquare'] == 0].copy()

# Calculate isolated averages, with reliable fallbacks
avgFeatureTimes = {}
fallbackTimes = {
    'isSquare\nMatrix Archetype': 0.01,
    'Diagonally Dominant\nRow Fraction': 0.15,
    'Directional Mean Bias': 0.22,
    'Signed Frobenius Ratio': 0.22,
    'Brauer Max Product': 0.45,
    'Degeneracy Multiplier': 0.45,
    'Topological Entropy': 0.67,
    'RCM Bandwidth': 1.25,
    'Fiedler Value': 20.50
}

allExpectedFeatures = stage1Features + stage2Features + stage3Features + ['isSquare\nMatrix Archetype']
for feat in allExpectedFeatures:
    if not dfTiming.empty and feat in dfTiming.columns:
        avgFeatureTimes[feat] = dfTiming[feat].mean()
    else:
        avgFeatureTimes[feat] = fallbackTimes.get(feat, 0.5)

for metric in accuracyMetricsSquare:
    metric['Model Shape'] = 'Square'
for metric in accuracyMetricsRect:
    metric['Model Shape'] = 'Rectangular'

accuracyMetrics = accuracyMetricsSquare + accuracyMetricsRect
dfAccuracy = pd.DataFrame(accuracyMetrics)

stageCounts = evalData['computationStageRequired'].value_counts().sort_index()

figPerf = plt.figure(figsize=(36, 26))
gsPerf = plt.GridSpec(3, 2, height_ratios=[1, 1, 1.5], wspace=0.2, hspace=0.3)

linePalette = {
    1: 'dodgerblue',        # L1: Diag Dom
    2: 'darkorange',        # L2: Dir Mean Bias, Frobenius
    3: 'gold',              # L3: Brauer, Degeneracy
    4: 'mediumseagreen',    # L4: Topo Entropy (t2+t3)
    5: 'darkorchid',        # L5: RCM Bandwidth
    6: 'crimson'            # L6: Fiedler
}

boxPalette = {
    0: '#f1f5f9', # L0: Slate Gray
    1: '#dbeafe', # L1: Light Blue
    2: '#ffedd5', # L2: Light Orange
    3: '#fef08a', # L3: Light Yellow
    4: '#dcfce7', # L4: Light Green
    5: '#f3e8ff', # L5: Light Purple
    6: '#fee2e2'  # L6: Light Red
}

costMap = {
    'isSquare\nMatrix Archetype': 0,
    'Density': 0,
    'Diagonally Dominant\nRow Fraction': 1,
    'Directional Mean Bias': 2,
    'Signed Frobenius Ratio': 2,
    'Brauer Max Product': 3,
    'Brauer Mean Product': 3,
    'Brauer Min Product': 3,
    'Brauer Ratio' : 3,
    'Degeneracy Multiplier': 3,
    'Brauer Mean Product (Top)': 3,
    'Topological Entropy': 4,
    'RCM Bandwidth': 5,
    'RCM Compression Ratio': 5,
    'Fiedler Value': 6
}

def wrapTextEveryTwoWords(textData):
    if pd.isna(textData): return "Unknown"
    cleanText = str(textData).replace('\n', ' ')
    wordList = cleanText.split()
    return "\n".join([" ".join(wordList[i:i+2]) for i in range(0, len(wordList), 2)])

allRoutingFeatures = list(set(stage1Features + stage2Features + stage3Features))
labelEnc = LabelEncoder()
labelEnc.fit(evalData['assignedSolver'].astype(str))

@cacheDir.cache
def trainAndExtractSubtree(dataSubset, maxDepthSetting, matrixShape):
    if len(dataSubset) == 0: return None
    solverCounts = dataSubset['assignedSolver'].value_counts()
    validClasses = solverCounts[solverCounts >= 2].index
    validSubset = dataSubset[dataSubset['assignedSolver'].isin(validClasses)].copy()
    if len(validSubset) == 0: return None
    activeFeatureNames = [f for f in allRoutingFeatures if f not in ['isSquare', 'isSquare\nMatrix Archetype']]
    xSurrogate = validSubset[activeFeatureNames].copy()

    for colName in activeFeatureNames:
        safeCol = pd.to_numeric(xSurrogate[colName], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(-1e5)
        xSurrogate[colName] = np.clip(safeCol, a_min=-1e35, a_max=1e35)

    yEncoded = labelEnc.transform(validSubset['assignedSolver'].astype(str))

    if len(np.unique(yEncoded)) <= 1:
        className = labelEnc.inverse_transform([yEncoded[0]])[0] if len(yEncoded) > 0 else "Unknown"
        return {'type': 'leaf', 'class': className, 'samples': len(validSubset)}

    baseTreePath = DecisionTreeClassifier(random_state=42)
    ccpPath = baseTreePath.cost_complexity_pruning_path(xSurrogate, yEncoded)

    rawAlphas = ccpPath.ccp_alphas
    validAlphas = np.unique(rawAlphas[rawAlphas >= 0])
    if len(validAlphas) > 20: validAlphas = np.quantile(validAlphas, np.linspace(0, 1, 20))
    
    paramGrid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(2, 8, 1)),
        'max_features': [None, 'sqrt', 'log2', 0.8],
        'max_leaf_nodes': [None, 10, 20, 30, 40],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15],
        'min_impurity_decrease': [0.0, 0.005, 0.01],
        'ccp_alpha': validAlphas
    }

    if matrixShape == 'Rect':
  
        paramGrid = {
            'criterion': ['gini'],
            'max_depth': [3],
            'max_features': [None],
            'max_leaf_nodes': [None],
            'min_samples_split': [10],
            'min_samples_leaf': [5],
            'min_impurity_decrease': [0.0],
            'ccp_alpha': [0.0]
        }
        
    baseTree = DecisionTreeClassifier(random_state=42)
    cvSplit = max(3, min(20,len(validSubset)//5))
    cvStrategy = StratifiedShuffleSplit(n_splits=cvSplit, test_size=0.2, random_state=42)
    grid = GridSearchCV(baseTree, paramGrid, cv=cvStrategy, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
    grid.fit(xSurrogate, yEncoded)
    surrogateModel = grid.best_estimator_
    treeData = surrogateModel.tree_
    classNames = labelEnc.classes_

    print(f"CONSOLE LOG: Optimal Subtree Found! Parameters: {grid.best_params_}")
    print(f"CONSOLE LOG: Best CV Balanced Accuracy: {grid.best_score_:.4f}")

    def extractNode(nodeId):
        nodeClassIdx = np.argmax(treeData.value[nodeId][0])
        dominantClass = classNames[nodeClassIdx]
        if treeData.children_left[nodeId] == -1:
            return {'type': 'leaf', 'class': dominantClass, 'samples': treeData.n_node_samples[nodeId]}
        leftNode = extractNode(treeData.children_left[nodeId])
        rightNode = extractNode(treeData.children_right[nodeId])
        if leftNode['type'] == 'leaf' and rightNode['type'] == 'leaf' and leftNode['class'] == rightNode['class']:
            return {'type': 'leaf', 'class': leftNode['class'], 'samples': leftNode['samples'] + rightNode['samples']}
        return {
            'type': 'internal',
            'feature': activeFeatureNames[treeData.feature[nodeId]],
            'threshold': treeData.threshold[nodeId],
            'samples': treeData.n_node_samples[nodeId],
            'class': dominantClass,
            'left': leftNode,
            'right': rightNode
        }
    return extractNode(0)

def expandArbitraryPrecisionTarget(row, targetCol):
    if row[targetCol] == 'Route to Arbitrary Precision Arithmetic':
        numRows = pd.to_numeric(row.get('Num Rows', 0), errors='coerce')
        
        # Simple size threshold: Small goes to Dense, Large goes to Iterative
        if numRows < 2000: 
            return 'Route to Arbitrary Precision LU (MPFR)'
        else: 
            return 'Route to Arbitrary Precision GMRES (MPFR)'
            
    return row[targetCol]

def truncateTree(nodeDict, currentDepth, maxDepth):
    if nodeDict is None: return None
    # If it is already a leaf, or we hit our visual threshold, convert to leaf
    if nodeDict['type'] == 'leaf' or currentDepth >= maxDepth:
        return {'type': 'leaf', 'class': nodeDict.get('class', 'Unknown'), 'samples': nodeDict['samples']}
    return {
        'type': 'internal',
        'feature': nodeDict['feature'],
        'threshold': nodeDict['threshold'],
        'samples': nodeDict['samples'],
        'class': nodeDict.get('class', 'Unknown'),
        'left': truncateTree(nodeDict['left'], currentDepth + 1, maxDepth),
        'right': truncateTree(nodeDict['right'], currentDepth + 1, maxDepth)
    }

dfSquare = evalData[evalData['isSquare'] == True].copy()
dfRectangular = evalData[evalData['isSquare'] == False].copy()

print("CONSOLE LOG: Training and Extracting Deep Rectangular Tree...")
deepRectTree = trainAndExtractSubtree(dfRectangular, 10, 'Rect')

print("CONSOLE LOG: Training and Extracting Deep Square Tree...")
deepSquareTree = trainAndExtractSubtree(dfSquare, 10, 'Square')

# 1. The Mathematical Engine (Depth 10)
deepGlobalTree = {
    'type': 'internal',
    'feature': 'isSquare\nMatrix Archetype',
    'threshold': 0.5,
    'samples': len(evalData),
    'left': deepRectTree,
    'right': deepSquareTree
}

# 2. The Visual Representation
visualGlobalTree = {
    'type': 'internal',
    'feature': 'isSquare\nMatrix Archetype',
    'threshold': 0.5,
    'samples': len(evalData),
    'left': truncateTree(deepRectTree, 1, 4),
    'right': truncateTree(deepSquareTree, 1, 6)
}

def injectArbitraryPrecisionRules(nodeDict, dfSubset):
    if nodeDict is None: return None
    
    # Hunt for the surrogate's AP leaf
    if nodeDict['type'] == 'leaf' and 'Arbitrary Precision' in str(nodeDict.get('class', '')):
        apData = dfSubset[dfSubset['isSvdFailed'] == 1]
        
        # Base Case: No SVD Failures at all in this branch
        if len(apData) == 0:
            return {'type': 'leaf', 'class': 'Arbitrary Precision', 'samples': 0}
            
        denseData = apData[apData['Num Rows'] < 2000]
        remData = apData[apData['Num Rows'] >= 2000]
        
        denseCount = len(denseData)
        remCount = len(remData)
        
        # Pruning Check 1: If everything is Dense, collapse to a single leaf
        if remCount == 0:
            return {'type': 'leaf', 'class': 'Dense Factorization', 'samples': denseCount}
            
        lapackCount = len(remData[remData['Density'] > 0.05])
        iterCount = len(remData[remData['Density'] <= 0.05])
        
        # Build the Density Split (Right Side)
        if lapackCount == 0 and iterCount == 0:
             rightNode = None # Failsafe
        elif lapackCount == 0:
             # Pruning Check 2: If no MPLAPACK, collapse Density split to Iterative
             rightNode = {'type': 'leaf', 'class': 'GMRES / MPFR', 'samples': iterCount}
        elif iterCount == 0:
             # Pruning Check 3: If no Iterative, collapse Density split to MPLAPACK
             rightNode = {'type': 'leaf', 'class': 'MPLAPACK', 'samples': lapackCount}
        else:
             # Standard Density Split
             rightNode = {
                 'type': 'internal',
                 'feature': 'Density',
                 'threshold': 0.05,
                 'samples': remCount,
                 'class': 'Arbitrary Precision',
                 'left': {'type': 'leaf', 'class': 'GMRES / MPFR', 'samples': iterCount},
                 'right': {'type': 'leaf', 'class': 'MPLAPACK', 'samples': lapackCount}
             }
             
        # Pruning Check 4: If no Dense matrices, skip the Num Rows split entirely
        if denseCount == 0:
            return rightNode
            
        # Full Unpruned Split
        return {
            'type': 'internal',
            'feature': 'Num Rows',
            'threshold': 1999.9,
            'samples': len(apData),
            'class': 'Arbitrary Precision',
            'left': {'type': 'leaf', 'class': 'Dense Factorization', 'samples': denseCount},
            'right': rightNode
        }
    
    # Continue traversing the tree
    if nodeDict['type'] == 'internal':
        nodeDict['left'] = injectArbitraryPrecisionRules(nodeDict['left'], dfSubset)
        nodeDict['right'] = injectArbitraryPrecisionRules(nodeDict['right'], dfSubset)
        
    return nodeDict

# Execute the injection immediately after defining visualGlobalTree
visualGlobalTree = injectArbitraryPrecisionRules(visualGlobalTree, evalData)

# ---------------------------------------------------------
# PANEL 1: Rule Accuracy Heatmap (Row 1, Col 1)
# ---------------------------------------------------------
axAcc = figPerf.add_subplot(gsPerf[0, 0])
for metric in accuracyMetricsSquare:
    metric['Model Shape'] = 'Square'
for metric in accuracyMetricsRect:
    metric['Model Shape'] = 'Rectangular'

accuracyMetrics = accuracyMetricsSquare + accuracyMetricsRect
dfAccuracy = pd.DataFrame(accuracyMetrics)
if not dfAccuracy.empty:
    accPivot = dfAccuracy.set_index('Target Constraint')[['Balanced Acc', 'MCC', 'F1 Score']]
    annotAcc = dfAccuracy.set_index('Target Constraint')[['Gating Feature']]
    sns.heatmap(accPivot, annot=np.tile(annotAcc['Gating Feature'].values[:, None], (1, 3)), fmt="", cmap='viridis',
                cbar_kws={'label': 'Metric Score'}, linewidths=1, linecolor='white', ax=axAcc,
                annot_kws={"size": 11, "weight": "bold"}, vmin=0, vmax=1)
    axAcc.set_title("Single-Stump Rule Accuracy by Target\n(Gating Features)", fontsize=16, fontweight='bold')
    axAcc.set_ylabel("Solver Constraint", fontsize=14)

# ---------------------------------------------------------
# PANEL 2: Density vs Timing Line Chart (Row 1, Col 2)
# ---------------------------------------------------------
axLine = figPerf.add_subplot(gsPerf[0, 1])
axLine.set_title("Feature Compute Time vs. Matrix Density\n(Categorized by Topology Tier)", fontsize=16, fontweight='bold')
axLine.set_xlabel("Density (Nonzeros / (Rows * Cols))", fontsize=14)
axLine.set_ylabel("Compute Time (ms)", fontsize=14)
axLine.grid(True, linestyle=':', alpha=0.6)

if not dfTiming.empty:
    dfLine = dfTiming.sort_values(by='Density')
    # Plotting representative variables for each level
    axLine.plot(dfLine['Density'], dfLine['Diagonally Dominant\nRow Fraction'], label='L1: Diag Dominance (t1)', color=linePalette[1], linewidth=2)
    axLine.plot(dfLine['Density'], dfLine['Directional Mean Bias'], label='L2: Directional Bias (t2)', color=linePalette[2], linewidth=2)
    axLine.plot(dfLine['Density'], dfLine['Brauer Max Product'], label='L3: Brauer Product (t3)', color=linePalette[3], linewidth=2)
    axLine.plot(dfLine['Density'], dfLine['Topological Entropy'], label='L4: Topo Entropy (t2+t3)', color=linePalette[4], linewidth=2)
    axLine.plot(dfLine['Density'], dfLine['RCM Bandwidth'], label='L5: RCM Bandwidth (t5)', color=linePalette[5], linewidth=2)
    axLine.plot(dfLine['Density'], dfLine['Fiedler Value'], label='L6: Fiedler Value (t6)', color=linePalette[6], linewidth=2)
    axLine.set_yscale('log')
    axLine.legend(loc='upper left', fontsize=11)

# ---------------------------------------------------------
# PANEL 3: Deep Cumulative Computation Waterfall Heatmap (Depth 10)
# ---------------------------------------------------------
axHeatmap = figPerf.add_subplot(gsPerf[1, :])

def getTreeDepth(nodeDict):
    if nodeDict is None or nodeDict['type'] == 'leaf': return 0
    return 1 + max(getTreeDepth(nodeDict['left']), getTreeDepth(nodeDict['right']))

actualDepth = getTreeDepth(deepGlobalTree)
targetDepth = 10 if actualDepth >= 10 else actualDepth
totalPaths = 2 ** targetDepth
costMatrix = np.zeros((targetDepth + 1, totalPaths))

def buildCostHeatmap(nodeDict, currentDepth, colStart, colEnd, accumulatedCost, activeFeatures):
    if currentDepth > targetDepth: return
    if nodeDict is None: return
    if nodeDict['type'] == 'internal':
        featName = nodeDict['feature']
        if featName not in activeFeatures:
            timeCost = avgFeatureTimes.get(featName, 0.0)
            accumulatedCost += timeCost
            activeFeatures.add(featName)
        costMatrix[currentDepth, colStart:colEnd] = accumulatedCost
        midPoint = (colStart + colEnd) // 2
        buildCostHeatmap(nodeDict['left'], currentDepth + 1, colStart, midPoint, accumulatedCost, activeFeatures.copy())
        buildCostHeatmap(nodeDict['right'], currentDepth + 1, midPoint, colEnd, accumulatedCost, activeFeatures.copy())
    else:
        costMatrix[currentDepth:, colStart:colEnd] = accumulatedCost

buildCostHeatmap(deepGlobalTree, 0, 0, totalPaths, 0.0, set())

# Restore SymLogNorm to dynamically scale the heatmap so Fiedler doesn't wash out the rest
sns.heatmap(costMatrix, cmap='YlOrRd', ax=axHeatmap, norm=SymLogNorm(linthresh=1.0, vmin=0, vmax=costMatrix.max()),
            cbar_kws={'label': 'Cumulative Compute Penalty (ms) [Log Scale]'}, linewidths=0, xticklabels=False)
yLabels = [f"Depth {i}" for i in range(targetDepth + 1)]
axHeatmap.set_yticklabels(yLabels, rotation=0, fontsize=12)
axHeatmap.set_title(f"Accumulated Computational Penalty by Decision Path (Max Depth = {targetDepth})", fontsize=16, fontweight='bold')
axHeatmap.set_xlabel("Distinct Algorithmic Routing Branches", fontsize=14)

# ---------------------------------------------------------
# PANEL 4: Unified Visual Routing Flow Diagram
# ---------------------------------------------------------
axCombinedTree = figPerf.add_subplot(gsPerf[2, :])
axCombinedTree.set_axis_off()

def setNodeCoordinates(nodeDict, depthLvl, xOffset):
    if nodeDict is None: return xOffset
    if nodeDict['type'] == 'leaf':
        nodeDict['x'] = xOffset
        nodeDict['y'] = -depthLvl
        return xOffset + 2.5
    else:
        leftOffset = setNodeCoordinates(nodeDict['left'], depthLvl + 1, xOffset)
        rightOffset = setNodeCoordinates(nodeDict['right'], depthLvl + 1, leftOffset)
        nodeDict['x'] = (nodeDict['left']['x'] + nodeDict['right']['x']) / 2.0
        nodeDict['y'] = -depthLvl
        return rightOffset

setNodeCoordinates(visualGlobalTree, 0, 0.0)
totalGlobalSamples = len(evalData)

def drawCustomTree(nodeDict, ax):
    if nodeDict is None: return
    pctValue = (nodeDict['samples'] / totalGlobalSamples) * 100 if totalGlobalSamples > 0 else 0
    if nodeDict['type'] == 'leaf':
        abbrevClass = solverAbbrevDict.get(nodeDict['class'], nodeDict['class'])
        nodeText = f"{wrapTextEveryTwoWords(abbrevClass)}\n{pctValue:.1f}%"
        boxColor = '#fde047'
    else:
        wrappedFeature = wrapTextEveryTwoWords(nodeDict['feature'])
        if nodeDict['threshold'] < -10000: conditionText = "Computation Failed / Missing"
        else: conditionText = f"<= {nodeDict['threshold']:.1f}"
        nodeText = f"{wrappedFeature}\n{conditionText}\n{pctValue:.1f}%"
        nodeCost = costMap.get(nodeDict['feature'], 0)
        boxColor = boxPalette.get(nodeCost, '#ffffff')
    bboxProps = dict(boxstyle="round,pad=0.5", facecolor=boxColor, edgecolor="black", lw=1)
    ax.text(nodeDict['x'], nodeDict['y'], nodeText, ha="center", va="center", bbox=bboxProps, fontsize=9, zorder=3)
    if nodeDict['type'] == 'internal':
        leftX, leftY = nodeDict['left']['x'], nodeDict['left']['y']
        rightX, rightY = nodeDict['right']['x'], nodeDict['right']['y']
        ax.plot([nodeDict['x'], leftX], [nodeDict['y'], leftY], 'k-', zorder=1)
        ax.plot([nodeDict['x'], rightX], [nodeDict['y'], rightY], 'k-', zorder=1)
        ax.text((nodeDict['x'] + leftX) / 2, (nodeDict['y'] + leftY) / 2, r'$\leq$ True',
                ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='darkgreen', alpha=0.9), zorder=2)
        ax.text((nodeDict['x'] + rightX) / 2, (nodeDict['y'] + rightY) / 2, r'$>$ False',
                ha='center', va='center', fontsize=9, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='darkred', alpha=0.9), zorder=2)
        drawCustomTree(nodeDict['left'], ax)
        drawCustomTree(nodeDict['right'], ax)

drawCustomTree(visualGlobalTree, axCombinedTree)

allX, allY = [], []
def getBounds(n):
    if n is None: return
    allX.append(n['x'])
    allY.append(n['y'])
    if n['type'] == 'internal':
        getBounds(n['left'])
        getBounds(n['right'])
getBounds(visualGlobalTree)
if allX and allY:
    axCombinedTree.set_xlim(min(allX) - 1.5, max(allX) + 1.5)
    axCombinedTree.set_ylim(min(allY) - 0.5, max(allY) + 0.5)

axCombinedTree.set_title("Unified Global Agentic Solver Decision Tree (Colored by Compute Level)", fontsize=18, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Pipeline Replicator
def engineerRoutingFeatures(df, brauerCols=None):
    if brauerCols is None: brauerCols = ['Brauer Min Product', 'Brauer Max Product', 'Brauer Max Center Distance']
    dfClean = df.copy()
    if 'Num Rows' in dfClean.columns and 'Num Cols' in dfClean.columns:
        dfClean['isSquare'] = (dfClean['Num Rows'] == dfClean['Num Cols']).astype(int)
    for stringCol in ['Full Numerical Rank?', 'Cholesky Candidate', 'Condition Number', 'Minimum Singular Value', 'Positive Definite']:
        if stringCol in dfClean.columns and dfClean[stringCol].dtype == object:
            dfClean[stringCol] = dfClean[stringCol].replace({'yes': 1, 'Yes': 1, '1': 1, 'no': 0, 'No': 0, '0': 0})
    if 'Condition Number' in dfClean.columns and 'Minimum Singular Value' in dfClean.columns:
        condCol = pd.to_numeric(dfClean['Condition Number'], errors='coerce')
        msvCol = pd.to_numeric(dfClean['Minimum Singular Value'], errors='coerce')
        dfClean['isSvdFailed'] = (condCol.isna() | np.isinf(condCol) | (condCol >= 1e15) | msvCol.isna()).astype(int)
    if 'Num Dmperm Blocks' in dfClean.columns:
        blockCol = pd.to_numeric(dfClean['Num Dmperm Blocks'], errors='coerce')
        dfClean['isIrreducible'] = ((blockCol <= 1) | blockCol.isna()).astype(int)
        dfClean['isDecomposable'] = (blockCol >= 5).astype(int)
    rankCol = None
    if 'Full Numerical Rank?' in dfClean.columns:
        rankCol = pd.to_numeric(dfClean['Full Numerical Rank?'], errors='coerce')
        dfClean['Rank Collapse'] = np.where(rankCol.isna(), np.nan, (rankCol == 0).astype(int))
    if 'Cholesky Candidate' in dfClean.columns:
        cholCol = pd.to_numeric(dfClean['Cholesky Candidate'], errors='coerce')
        dfClean['isCholesky'] = (cholCol == 1).astype(int)
    if set(['Nonzeros', 'Num Rows', 'Num Cols']).issubset(dfClean.columns):
        dfClean['Density'] = dfClean['Nonzeros'].astype(float) / (dfClean['Num Rows'].astype(float) * dfClean['Num Cols'].astype(float))
    if set(['RCM Bandwidth', 'Num Rows']).issubset(dfClean.columns):
        dfClean['RCM Compression Ratio'] = dfClean['RCM Bandwidth'] / dfClean['Num Rows']
    if set(['RCM Compression Ratio', 'Density']).issubset(dfClean.columns):    
        dfClean['Topological Entropy'] = dfClean['RCM Compression Ratio'] / (dfClean['Density'] + 1e-10)
    if set(['Brauer Max Center Distance', 'Directional Mean Bias']).issubset(dfClean.columns):
        dfClean['Degeneracy Multiplier'] = dfClean['Brauer Max Center Distance'] * dfClean['Directional Mean Bias']
    if set(['Brauer Min Product', 'Brauer Max Product']).issubset(dfClean.columns):
        maxProd = pd.to_numeric(dfClean['Brauer Max Product'], errors='coerce')
        minProd = pd.to_numeric(dfClean['Brauer Min Product'], errors='coerce').fillna(0)
        dfClean['Brauer Ratio'] = (np.sqrt(maxProd) / (np.sqrt(minProd + 1e-10))).replace([np.inf, -np.inf], 1e35)
        dfClean['Brauer Ratio'] = np.clip(dfClean['Brauer Ratio'], a_min=1e-35, a_max=1e35)
    for baseCol in brauerCols:
        if baseCol in dfClean.columns:
            safeNumericCol = pd.to_numeric(dfClean[baseCol], errors='coerce').fillna(0)
            dfClean[baseCol] = np.sqrt(np.maximum(safeNumericCol, 0))
    if 'sprank(A)-rank(A)' in dfClean.columns:
        degCol = pd.to_numeric(dfClean['sprank(A)-rank(A)'], errors='coerce')
        dfClean['isDegenerate'] = np.where(degCol.isna(), np.nan, (degCol > 0).astype(int))
    return dfClean

# ============================
# FINAL VALIDATION & OVERLAY
# ============================
targetRoutingGroup = 'trueAssignedSolver'

def prepareFullTestData(dfTrain, dfTest, originalBaseCols):
    baseColsOnly = [c for c in originalBaseCols if not str(c).startswith('missingindicator_')]
    trainPrep = dfTrain[baseColsOnly].copy()
    testPrep = dfTest[baseColsOnly].copy()

    for df in [trainPrep, testPrep]:
        df.replace([np.inf, -np.inf], [1e35, -1e35], inplace=True)
        if 'Minimum Singular Value' in df.columns: df['Minimum Singular Value'] = df['Minimum Singular Value'].fillna(0.0)
        if 'Condition Number' in df.columns: df['Condition Number'] = df['Condition Number'].fillna(1e35)
        for colName in baseColsOnly:
            safeCol = pd.to_numeric(df[colName], errors='coerce').astype(float)
            df[colName] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
    masterImputer = SimpleImputer(strategy='median', add_indicator=True)
    trainImputed = masterImputer.fit_transform(trainPrep)
    testImputed = masterImputer.transform(testPrep)
    expandedColNames = masterImputer.get_feature_names_out(baseColsOnly)
    masterScaler = StandardScaler().fit(trainImputed)
    trainScaledDf = pd.DataFrame(masterScaler.transform(trainImputed), columns=expandedColNames)
    testScaledDf = pd.DataFrame(masterScaler.transform(testImputed), columns=expandedColNames)
    return trainScaledDf, testScaledDf

try: 
    dfTestRaw = pd.read_csv('testMatrices.csv')
    dfTestRaw.rename(columns={'Strictly Diagonally Dominant Row Fraction': 'Diagonally Dominant\nRow Fraction'}, inplace=True)
except FileNotFoundError: raise FileNotFoundError("CRITICAL ERROR: 'testMatrices.csv' not found.")

dfTrainEngineered = dfTransformed.copy()
dfTestEngineered = engineerRoutingFeatures(dfTestRaw, brauerCols=['Brauer Min Product', 'Brauer Max Product', 'Brauer Max Center Distance'])

# Calculate Ground Truths and Predictions
dfTestEngineered[targetRoutingGroup] = dfTestEngineered.apply(getGroundTruthSolver, axis=1)
routerOutputsTest = dfTestEngineered.apply(agenticSolverRouter, axis=1)
dfTestEngineered['Predicted Target'] = [res[0] for res in routerOutputsTest]
dfTestEngineered['Classification Result'] = np.where(dfTestEngineered[targetRoutingGroup] == dfTestEngineered['Predicted Target'], 'Correct', 'Incorrect')

# 1. Prepare Full DataFrames for Shift Analysis & PCA
allRoutingFeatures = list(set(stage1Features + stage2Features + stage3Features))
trainScaledDfFull, testScaledDfFull = prepareFullTestData(dfTrainEngineered, dfTestEngineered, allRoutingFeatures)

# 2. Reconstruct the Untargeted Kernel PCA Projection
targetPCACols = bestResults['PCA']['prunedCols']
trainRawForPCA = dfTrainEngineered[targetPCACols].copy()
testRawForPCA = dfTestEngineered[targetPCACols].copy()

for df in [trainRawForPCA, testRawForPCA]:
    df.replace([np.inf, -np.inf], [1e35, -1e35], inplace=True)
    if 'Minimum Singular Value' in df.columns: df['Minimum Singular Value'] = df['Minimum Singular Value'].fillna(0.0)
    if 'Condition Number' in df.columns: df['Condition Number'] = df['Condition Number'].fillna(1e35)
    for colName in targetPCACols:
        safeCol = pd.to_numeric(df[colName], errors='coerce').astype(float)
        df[colName] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
pcaImputer = SimpleImputer(strategy='median', add_indicator=True)
trainImpPCA = pcaImputer.fit_transform(trainRawForPCA)
testImpPCA = pcaImputer.transform(testRawForPCA)
pcaScaler = MinMaxScaler()
trainScaledPCA = pcaScaler.fit_transform(trainImpPCA)
testScaledPCA = pcaScaler.transform(testImpPCA)
pcaModelFinal = KernelPCA(n_components=2, kernel='cosine', random_state=42).fit(trainScaledPCA)
trainPCAEmb = pcaModelFinal.transform(trainScaledPCA)
testPCAEmb = pcaModelFinal.transform(testScaledPCA)
dfTrainEngineered['final_PCA_1'], dfTrainEngineered['final_PCA_2'] = trainPCAEmb[:, 0], trainPCAEmb[:, 1]
dfTestEngineered['final_PCA_1'], dfTestEngineered['final_PCA_2'] = testPCAEmb[:, 0], testPCAEmb[:, 1]

# Combine Sets for Confusion Matrix & ROC
dfCombinedFull = pd.concat([dfTrainEngineered, dfTestEngineered], ignore_index=True)
dfCombinedFull[targetRoutingGroup] = dfCombinedFull.apply(getGroundTruthSolver, axis=1)
routerOutputsComb = dfCombinedFull.apply(agenticSolverRouter, axis=1)
dfCombinedFull['Predicted Target'] = [res[0] for res in routerOutputsComb]

# Expand the unified predictions back into specific rules for the heatmap
dfCombinedFull[targetRoutingGroup] = dfCombinedFull.apply(lambda r: expandArbitraryPrecisionTarget(r, targetRoutingGroup), axis=1)
dfCombinedFull['Predicted Target'] = dfCombinedFull.apply(lambda r: expandArbitraryPrecisionTarget(r, 'Predicted Target'), axis=1)

# Now your existing abbreviation mapping will catch the expanded routes normally:
dfCombinedFull['Abbrev Truth'] = dfCombinedFull[targetRoutingGroup].map(solverAbbrevDict).fillna(dfCombinedFull[targetRoutingGroup])
dfCombinedFull['Abbrev Pred'] = dfCombinedFull['Predicted Target'].map(solverAbbrevDict).fillna(dfCombinedFull['Predicted Target'])

knownClasses = list(solverAbbrevDict.keys())
yProbaComb = np.zeros((len(dfCombinedFull), len(knownClasses)))
for idx, predClass in enumerate(dfCombinedFull['Predicted Target']):
    if predClass in knownClasses: yProbaComb[idx, knownClasses.index(predClass)] = 1.0
yTestBinComb = label_binarize(dfCombinedFull[targetRoutingGroup], classes=knownClasses)

# ============================
# FINAL VALIDATION PLOTTING
# ============================
figVal, axesVal = plt.subplots(2, 2, figsize=(22, 18))
figVal.suptitle("Final Validation: Core Constraints & Targeted Kernel-PCA Topology Overlay", fontsize=22, fontweight='bold')

# --- TOP LEFT: Distribution Shift (Cohen's d style for ALL used features) ---
shiftData = []
for feat in allRoutingFeatures:
    if feat in trainScaledDfFull.columns:
        train_vals = trainScaledDfFull[feat].values
        test_vals = testScaledDfFull[feat].values
        mean_diff = np.mean(test_vals) - np.mean(train_vals)
        shiftData.append({'Feature': feat, 'Mean Shift (Std Devs)': mean_diff})

dfShift = pd.DataFrame(shiftData).sort_values('Mean Shift (Std Devs)', key=abs, ascending=False).head(15)

sns.barplot(data=dfShift, x='Mean Shift (Std Devs)', y='Feature', ax=axesVal[0, 0], palette='coolwarm')
axesVal[0, 0].axvline(0, color='black', linewidth=2)
axesVal[0, 0].set_title("Distribution Shift of AI Routing Predictors\n(Test vs. Train Set)", fontsize=14, fontweight='bold')
axesVal[0, 0].set_xlabel("Mean Shift (Standard Deviations)", fontsize=12)
axesVal[0, 0].set_ylabel("")

# --- TOP RIGHT: KernelPCA Overlay ---
axesVal[0, 1].scatter(dfTrainEngineered['final_PCA_1'], dfTrainEngineered['final_PCA_2'], color='lightgrey', s=35, alpha=0.5, label='Training Background')

correctMask = dfTestEngineered['Classification Result'] == 'Correct'
incorrectMask = dfTestEngineered['Classification Result'] == 'Incorrect'

axesVal[0, 1].scatter(dfTestEngineered.loc[correctMask, 'final_PCA_1'], dfTestEngineered.loc[correctMask, 'final_PCA_2'], c='blue', marker='o', s=100, alpha=0.7, edgecolor='white', label='Test Correctly Routed')
axesVal[0, 1].scatter(dfTestEngineered.loc[incorrectMask, 'final_PCA_1'], dfTestEngineered.loc[incorrectMask, 'final_PCA_2'], c='red', marker='X', s=120, alpha=0.8, edgecolor='black', label='Test Misrouted')

axesVal[0, 1].set_title("Test Set Routing Accuracy on Targeted KernelPCA Overlay", fontsize=14, fontweight='bold')
axesVal[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# --- BOTTOM LEFT: Core Constraint ROC-AUC ---
coreConditions = ['isSvdFailed', 'Rank Collapse', 'Positive Definite', 'isCholesky', 'isIrreducible', 'isDecomposable', 'isDegenerate']

# Map raw variables to clean display names
cleanLabelMap = {
    'isSvdFailed': 'SVD Failed',
    'Rank Collapse': 'Rank Collapse',
    'Positive Definite': 'Positive Definite',
    'isCholesky': 'Cholesky Candidate',
    'isIrreducible': 'Irreducible',
    'isDecomposable': 'Decomposable',
    'isDegenerate': 'Degenerate'
}

cmap = plt.get_cmap("tab10")

for i, condition in enumerate(coreConditions):
    true_labels, pred_probs = [], []
    for _, row in dfCombinedFull.iterrows():
        true_val = row.get(condition)
        if pd.isna(true_val): continue
        _, prob = evaluate_deep_rule(row, condition)
        true_labels.append(int(true_val > 0))
        pred_probs.append(prob)
        
    if len(true_labels) > 0 and len(np.unique(true_labels)) > 1:
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        rocAuc = auc(fpr, tpr)
        cleanName = cleanLabelMap.get(condition, condition)
        axesVal[1, 0].plot(fpr, tpr, lw=2, color=cmap(i % 10), label=f'{cleanName} (AUC = {rocAuc:.2f})')

axesVal[1, 0].plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
axesVal[1, 0].set_xlim([0.0, 1.0])
axesVal[1, 0].set_ylim([0.0, 1.05])
axesVal[1, 0].set_xlabel('False Positive Rate', fontsize=12)
axesVal[1, 0].set_ylabel('True Positive Rate', fontsize=12)
axesVal[1, 0].set_title('ROC-AUC: Core AI Constraint Predictions (Combined Sets)', fontsize=14, fontweight='bold')
axesVal[1, 0].legend(loc="lower right", fontsize=10)

# --- BOTTOM RIGHT: Heatmap ---
# Ensure abbreviation mappings exist for the test set explicitly
dfTestEngineered['Abbrev Truth'] = dfTestEngineered[targetRoutingGroup].map(solverAbbrevDict).fillna(dfTestEngineered[targetRoutingGroup])
dfTestEngineered['Abbrev Pred'] = dfTestEngineered['Predicted Target'].map(solverAbbrevDict).fillna(dfTestEngineered['Predicted Target'])

# Generate both confusion matrices
cmComb = confusion_matrix(dfCombinedFull['Abbrev Truth'], dfCombinedFull['Abbrev Pred'], labels=list(solverAbbrevDict.values()))
cmTest = confusion_matrix(dfTestEngineered['Abbrev Truth'], dfTestEngineered['Abbrev Pred'], labels=list(solverAbbrevDict.values()))

cmDfComb = pd.DataFrame(cmComb, index=list(solverAbbrevDict.values()), columns=list(solverAbbrevDict.values()))
cmDfTest = pd.DataFrame(cmTest, index=list(solverAbbrevDict.values()), columns=list(solverAbbrevDict.values()))

# Clean up empty rows/columns for display
activeMask = (cmDfComb.sum(axis=1) != 0) | (cmDfComb.sum(axis=0) != 0)
cmDfComb = cmDfComb.loc[activeMask, activeMask]
cmDfTest = cmDfTest.loc[activeMask, activeMask]

cmNormComb = cmDfComb.div(cmDfComb.sum(axis=1), axis=0).fillna(0) * 100

# Build custom text annotations: "Total \n [Test]"
custom_annotations = np.empty_like(cmDfComb.values, dtype=object)
for i in range(cmDfComb.shape[0]):
    for j in range(cmDfComb.shape[1]):
        valComb = cmDfComb.iloc[i, j]
        valTest = cmDfTest.iloc[i, j]
        if valComb == 0:
            custom_annotations[i, j] = ""
        else:
            custom_annotations[i, j] = f"{valComb}"

sns.heatmap(cmNormComb, annot=custom_annotations, fmt="", cmap='Blues', 
            cbar_kws={'label': 'Accuracy (%)'}, annot_kws={"size": 11, "weight": "bold"}, ax=axesVal[1, 1])

axesVal[1, 1].set_ylabel('True Target Group', fontsize=12, fontweight='bold')
axesVal[1, 1].set_xlabel('Predicted Target Group', fontsize=12, fontweight='bold')
axesVal[1, 1].set_title('Agentic Router Confusion Matrix\nTotal Volume [Test Set Volume]', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()