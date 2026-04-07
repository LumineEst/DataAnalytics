import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import time
import re
import xgboost as xgb
import shap
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as splinalg
from sklearn.preprocessing import StandardScaler, minmax_scale, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, precision_score, average_precision_score, f1_score, recall_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

dfMatrix = pd.read_csv('matrixdata.csv').set_index('Matrix ID')
dfTransforms = pd.read_csv('transforms.csv').set_index('Matrix ID')

dfMatrix.columns = dfMatrix.columns.str.strip()
dfTransforms.columns = dfTransforms.columns.str.strip()
dfMatrix = dfMatrix.loc[:, ~dfMatrix.columns.duplicated()]
dfTransforms = dfTransforms.loc[:, ~dfTransforms.columns.duplicated()]

# ==========================
# MATRIX GROUP LOGIC & MASKS
# ==========================
dfOriginal = dfMatrix.reset_index()
dfTransformed = dfTransforms.combine_first(dfMatrix).reset_index()

dfOriginal['isSquare'] = dfOriginal['Num Rows'] == dfOriginal['Num Cols']
dfTransformed['isSquare'] = dfTransformed['Num Rows'] == dfTransformed['Num Cols']
brauerCols = ['Brauer Mean Product', 'Brauer Min Product', 'Brauer Max Product', 'Brauer Mean Product (Top)']

# Computation of composite metrics
for df in [dfOriginal, dfTransformed]:
    if set(['Nonzeros', 'Num Rows', 'Num Cols', 'RCM Bandwidth']).issubset(df.columns):
        df['Density'] = df['Nonzeros'] / (df['Num Rows'] * df['Num Cols'])
        df['RCM Compression Ratio'] = df['RCM Bandwidth'] / df['Num Rows']
        df['Topological Entropy'] = df['RCM Compression Ratio'] / (df['Density'] + 1e-10)
    if set(['Brauer Max Center Distance', 'Directional Mean Bias']).issubset(df.columns):
        df['Degeneracy Multiplier'] = df['Brauer Max Center Distance'] * df['Directional Mean Bias']
    for baseCol in brauerCols:
        if baseCol in df.columns:
            safeNumericCol = pd.to_numeric(df[baseCol], errors='coerce').fillna(0)
            df[baseCol] = np.sqrt(np.maximum(safeNumericCol, 0))

def applyPositivityMask(df):
    mask = pd.Series(False, index=df.index)
    for col in ['Directional Mean Bias', 'Signed Frobenius Ratio']:
        if col in df.columns:
            mask |= np.isinf(pd.to_numeric(df[col], errors='coerce'))
        df['isInfinitePositivity'] = mask
    return df

dfOriginal = applyPositivityMask(dfOriginal)
dfTransformed = applyPositivityMask(dfTransformed)

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

# Handle structural rank degeneracy
if set(['Structural Rank', 'Rank']).issubset(dfOriginal.columns):
    dfOriginal['Rank Degeneracy'] = dfOriginal['Structural Rank'] - dfOriginal['Rank']
if set(['Structural Rank', 'Rank']).issubset(dfTransformed.columns):
    dfTransformed['Rank Degeneracy'] = dfTransformed['Structural Rank'] - dfTransformed['Rank'] 

dfSquareOriginal = dfOriginal[dfOriginal['isSquare'] == True].copy()
dfSquareTransformed = dfTransformed[dfTransformed['isSquare'] == True].copy()
dfRectangularTransformed = dfTransformed[dfTransformed['isSquare'] == False].copy()

for df in [dfSquareOriginal, dfSquareTransformed, dfRectangularTransformed, dfTransformed, dfOriginal]:
    if 'Full Numerical Rank?' in df.columns:
        df['Rank Collapse'] = (df['Full Numerical Rank?'] == 0).astype(int)

# ====================================================
# WILCOXON SIGNED-RANK TEST OF PRE-POST TRANSFORM DATA
# ====================================================
keys = ['Matrix ID', 'Name', 'Group', 'Group.1']
overlapCols = [c for c in dfMatrix.columns if c in dfTransforms.columns and c not in keys]
wilcoxonResults = []
if len(overlapCols) > 0:
    for col in overlapCols:
        pairedDf = pd.DataFrame({
            'Original': pd.to_numeric(dfSquareOriginal[col], errors='coerce'),
            'Transformed': pd.to_numeric(dfSquareTransformed[col], errors='coerce')
        }).replace([np.inf, -np.inf], np.nan).dropna()
        if len(pairedDf) > 0:
            stat, p = stats.wilcoxon(pairedDf['Original'], pairedDf['Transformed'])
            wilcoxonResults.append({'Metric': col, 'WStat': stat, 'PValue': p})
    dfWilcoxon = pd.DataFrame(wilcoxonResults).sort_values(by='PValue')
plt.figure(figsize=(10, max(4, len(dfWilcoxon) * 0.5)))
dfWilcoxon['LogP'] = -np.log10(dfWilcoxon['PValue'] + 1e-300)
sns.barplot(data=dfWilcoxon, x='LogP', y='Metric', palette='viridis')
plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05 Threshold')
plt.title('Wilcoxon Signed-Rank Test Significance (-log10 P-Value)')
plt.xlabel('-log10(P-Value)')
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# KRUSKAL-WALLIS H-TEST OF MATRIX GROUPINGS
# =========================================
hTestResults = []
configs = {'Square Transformed': dfSquareTransformed, 'Rectangular Transformed': dfRectangularTransformed}
masks = ['isInfinitePositivity', 'matrixGroup']
ignoreCols = keys + ['Kind', 'Type', 'Author', 'isSquare', 'isInfinitePositivity', 'matrixGroup', 'Gershgorin Discs']

for configName, df in configs.items():
    numCols = [c for c in df.columns if c not in ignoreCols and pd.api.types.is_numeric_dtype(df[c])]
    for mask in masks:
        uniqueGroups = df[mask].dropna().unique()
        if len(uniqueGroups) < 2: continue
        for metric in numCols:
            groupsData = []
            for g in uniqueGroups:
                data = df[df[mask] == g][metric].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) > 0: groupsData.append(data)
            if len(groupsData) >= 2:
                try:
                    stat, p = stats.kruskal(*groupsData)
                    hTestResults.append({'DataFrame': configName, 'SplitBy': mask, 'Metric': metric, 'HStat': stat, 'PValue': p})
                except Exception: pass
dfHTest = pd.DataFrame(hTestResults).sort_values(by='PValue')

# Visualize H-Test Significance via Heatmap
dfHTest['LogP'] = -np.log10(dfHTest['PValue'] + 1e-300)
labelMap = {'isInfinitePositivity': 'Positivity\nSkew', 'matrixGroup': 'Matrix\nGroup'}
dfHTest['SplitByShort'] = dfHTest['SplitBy'].map(labelMap)
dfSquare = dfHTest[dfHTest['DataFrame'] == 'Square Transformed']
dfRect = dfHTest[dfHTest['DataFrame'] == 'Rectangular Transformed']
fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(dfHTest['Metric'].unique()) * 0.4)))

# Subplot 1: Square Matrices
pivotSquareLogP = dfSquare.pivot_table(index='Metric', columns='SplitByShort', values='LogP', fill_value=0)
pivotSquarePVal = dfSquare.pivot_table(index='Metric', columns='SplitByShort', values='PValue', fill_value=1.0)
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
plt.suptitle('Kruskal-Wallis H-Test Significance by Core Matrix Groupings', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# =======================
# HIERARCHICAL CLUSTERMAP
# =======================

for df in [dfSquareOriginal, dfSquareTransformed, dfRectangularTransformed]:
    if 'Full Numerical Rank?' in df.columns:
        df['Rank Collapse'] = (df['Full Numerical Rank?'] == 0).astype(int)
    elif 'Rank Degeneracy' in df.columns:
        df['Rank Collapse'] = (df['Rank Degeneracy'] > 0).astype(int)

solvabilityTargets = [
    'Condition Number', 'Minimum Singular Value', 'Matrix Norm', 'Num Dmperm Blocks', 'Strongly Connect Components',
    'Rank Collapse', 'Positive Definite', 'Cholesky Candidate'
]

configs = [('Rectangular Transformed', dfRectangularTransformed), ('Square Original', dfSquareOriginal), ('Square Transformed', dfSquareTransformed)]
numTargets = len(solvabilityTargets)
numConfigs = len(configs)
transformCols = [c for c in dfTransforms.columns if pd.api.types.is_numeric_dtype(dfTransforms[c]) and c not in keys + solvabilityTargets + ['isSquare']]
masterPredictors = transformCols + ['Density', 'RCM Compression Ratio', 'Topological Entropy', 'Degeneracy Multiplier']

clusterCols = [c for c in masterPredictors if c in dfTransformed.columns]
groupedDf = dfTransformed.groupby(['matrixGroup', 'isInfinitePositivity', 'isSquare'])[clusterCols].median()
groupedDf = groupedDf.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
groupedDf.index = [f"{g[0]}\nPosInf: {g[1]}\nSq: {g[2]}" for g in groupedDf.index]
imputer = SimpleImputer(strategy='median')
imputedData = imputer.fit_transform(groupedDf)
imputedData = np.clip(imputedData, -1e35, 1e35)
scaler = StandardScaler()
scaledData = scaler.fit_transform(imputedData)
scaledDf = pd.DataFrame(scaledData, index=groupedDf.index, columns=groupedDf.columns)
scaledDf = scaledDf.loc[:, scaledDf.var() > 0.01].clip(lower=-5, upper=5)
plt.figure(figsize=(14, 10))
cg = sns.clustermap(scaledDf.T, cmap='coolwarm', metric='euclidean', method='ward', figsize=(16, 12), cbar_kws={'label': 'Z-Score'}, xticklabels=True, yticklabels=True)
cg.fig.suptitle("Hierarchical Clustermap of Metrics (Aggregated by 3 Masks)", fontsize=16, fontweight='bold', x=0.98, y=0.98, ha='right')
plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.show()

# ==========================================
# RANDOM FOREST & SUBPLOTTED PARALLEL COORDS
# ==========================================

xgbFeatures = list (masterPredictors)
for col in masterPredictors:
    if col in dfTransformed.columns:
        # Create explicit boolean signal columns
        infCol = f"{col}_is_inf"
        nanCol = f"{col}_is_missing"
        dfTransformed[infCol] = np.isinf(pd.to_numeric(dfTransformed[col], errors='coerce')).astype(int)
        dfTransformed[nanCol] = dfTransformed[col].isna().astype(int)
        # Convert infinites to NaN so XGBoost can natively handle them
        dfTransformed[col] = dfTransformed[col].replace([np.inf, -np.inf], np.nan)
        # Add the new boolean signals to our master predictor list
        xgbFeatures.extend([infCol, nanCol])

fig, axes = plt.subplots(nrows=numTargets * 2, ncols=numConfigs, figsize=(26, 80))

for tIdx, targetCol in enumerate(solvabilityTargets):
    for cIdx, (configName, dfConfig) in enumerate(configs):
        axPc = axes[tIdx * 2, cIdx]
        axBar = axes[tIdx * 2 + 1, cIdx]
        featureCols = [c for c in masterPredictors if c in dfConfig.columns]
        if 'isInfinitePositivity' in dfConfig.columns and 'isInfinitePositivity' not in featureCols: featureCols.append('isInfinitePositivity')
        pcaFeatures = ['Strictly Diagonally Dominant Row Fraction', 'Topological Entropy', 'Density', 'RCM Compression Ratio', 'Signed Frobenius Ratio']
        tsneFeatures = ['RCM Compression Ratio', 'Topological Entropy', 'Strictly Dominant Row Fraction', 'Brauer Min Product', 'Fiedler Value']
        umapFeatures = ['Degeneracy Multiplier', 'Brauer Min Product', 'Directional Mean Bias', 'Brauer Max Center Distance', 'Fiedler Value', 'Signed Frobenius Ratio']
        # Replace the 4 invalid Rectangular charts with custom Parallel Coordinates
        if configName == 'Rectangular Transformed' and targetCol in ['Positive Definite', 'Cholesky Candidate']:
            targetSequence = ['Rank Collapse', 'Minimum Singular Value', 'Matrix Norm', 'Condition Number', 'Num Dmperm Blocks', 'Strongly Connect Components', 'Cholesky Candidate']
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
            continue # Skip the Random Forest generation for these specific cells

        # ==================== DATA PREP ====================
        rfData = dfConfig[[targetCol, 'matrixGroup'] + featureCols].copy()
        if 'isInfinitePositivity' in rfData.columns:
            rfData['isInfinitePositivity'] = rfData['isInfinitePositivity'].astype(int)
        if targetCol not in ['matrixGroup', 'isInfinitePositivity']:
            rfData[targetCol] = pd.to_numeric(rfData[targetCol], errors='coerce').astype(float)
        for col in featureCols:
            if col != 'isInfinitePositivity':
                safeCol = pd.to_numeric(rfData[col], errors='coerce').astype(float)
                rfData[col] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
        rfData = rfData.replace([np.inf, -np.inf], [1e35, -1e35]).dropna()
        
        # ==================== MODEL TRAINING ====================
        X = rfData[featureCols].values
        is_classification = targetCol in ['Rank Collapse', 'Positive Definite', 'Cholesky Candidate']
        if is_classification:
            y = rfData[targetCol].astype(int).values
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42)
            cvFolds = KFold(n_splits=5, shuffle=True, random_state=42)
            cvScores = cross_val_score(rf, X, y, cv=cvFolds, scoring='accuracy')
            metricLabel = "Accuracy"
        else:
            if targetCol in ['Condition Number', 'Minimum Singular Value', 'Matrix Norm', 'Num Dmperm Blocks', 'Strongly Connect Components']:
                y = np.log1p(np.clip(rfData[targetCol].values, 0, 1e35))
            else:
                y = rfData[targetCol].values
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42)
            cvFolds = KFold(n_splits=5, shuffle=True, random_state=42)
            cvScores = cross_val_score(rf, X, y, cv=cvFolds, scoring='r2')
            metricLabel = "R²"
        accMean = np.mean(cvScores)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=featureCols).sort_values(ascending=False)
        topFeatures = importances.head(5).index.tolist()
        topImportances = importances.head(5).values

        pcData = rfData[topFeatures + [targetCol, 'matrixGroup']].copy()
        for col in topFeatures + [targetCol]:
            if col != 'isInfinitePositivity': pass
            pcData[col] = minmax_scale(pcData[col])
        renameMap = {col: col.replace(' ', '\n') for col in topFeatures + [targetCol]}
        pcData = pcData.rename(columns=renameMap)
        newTopFeatures = [renameMap[col] for col in topFeatures]
        newTargetCol = renameMap[targetCol]
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
        barLabels = [f.replace(' ', '\n') for f in topFeatures]
        
        sns.barplot(x=topImportances, y=barLabels, ax=axBar, palette='mako')
        accColor = 'darkred' if accMean < 0.5 else 'darkgreen'
        axBar.set_title(f"Model {metricLabel} (5-Fold CV): {accMean:.2f}", fontsize=14, fontweight='bold', color=accColor)
        axBar.set_xlabel("Relative Importance Contribution", fontsize=12)
        axBar.set_xlim(0, 1.0)
        if tIdx < numTargets - 1:
            axBar.axhline(y=len(barLabels), color='black', linewidth=4, alpha=0.5, clip_on=False)

plt.tight_layout()
plt.show()

# ========================
# DIMENSIONALITY REDUCTION
# ========================

targetGroups = ['Optimization', 'Applied Physics', 'Network Graphs']
dfDim = dfTransformed[dfTransformed['matrixGroup'].isin(targetGroups)].copy()
baseCols = [c for c in masterPredictors if c in dfDim.columns]
labels = dfDim['matrixGroup'].values
umapLabels = pd.factorize(labels)[0]
bestResults = {'PCA': {'score': -1}, 'TSNE': {'score': -1}, 'UMAP': {'score': -1}}

# Helper function to score subsets and find the best mathematical transformation
def score_subset(algo, cols):
    temp = dfDim[cols].replace([np.inf, -np.inf], np.nan).fillna(dfDim[cols].median())
    scalers = {
        'Standard': StandardScaler(),
        'Quantile': QuantileTransformer(output_distribution='uniform', random_state=42),
    }
    bestSc = -1
    bestScalerName = 'Standard'

    for sName, scaler in scalers.items():
        scaled = scaler.fit_transform(temp)
        try:
            if algo == 'PCA':
                emb = PCA(n_components=2, random_state=42).fit_transform(scaled)
            elif algo == 'TSNE':
                emb = TSNE(n_components=2, perplexity=min(30, len(temp)-1), random_state=42).fit_transform(scaled)
            elif algo == 'UMAP':
                emb = umap.UMAP(n_components=2, random_state=42).fit_transform(scaled.astype(float), y=umapLabels)
            sc = silhouette_score(emb, labels)
            if sc > bestSc:
                bestSc = sc
                bestScalerName = sName
        except: pass
    return bestSc, bestScalerName

# (Backward Elimination -> Forward Selection)
for algo in ['PCA', 'TSNE', 'UMAP']:
    currentCols = list(baseCols)
    bestScore, bestScaler = score_subset(algo, currentCols)
    
    # PASS 1: Backward Elimination
    improved = True
    while improved and len(currentCols) > 2:
        improved = False
        bestStepScore = -1
        colToDrop = None
        stepScaler = None
        for col in currentCols:
            testCols = [c for c in currentCols if c != col]
            sc, sName = score_subset(algo, testCols)
            if sc > bestStepScore:
                bestStepScore = sc
                colToDrop = col
                stepScaler = sName
        if bestStepScore > bestScore:
            bestScore = bestStepScore
            bestScaler = stepScaler
            currentCols.remove(colToDrop)
            improved = True
    # PASS 2: Forward Selection
    droppedCols = [c for c in baseCols if c not in currentCols]
    improved = True
    while improved and len(droppedCols) > 0:
        improved = False
        bestStepScore = -1
        colToAdd = None
        stepScaler = None
        for col in droppedCols:
            testCols = currentCols + [col]
            sc, sName = score_subset(algo, testCols)
            if sc > bestStepScore:
                bestStepScore = sc
                colToAdd = col
                stepScaler = sName
        if bestStepScore > bestScore:
            bestScore = bestStepScore
            bestScaler = stepScaler
            currentCols.append(colToAdd)
            droppedCols.remove(colToAdd)
            improved = True
    bestResults[algo] = {'prunedCols': currentCols, 'scalerName': bestScaler, 'score': bestScore}
    print(f"[{algo}] Pruned to {len(currentCols)} vars | Best Transform: {bestScaler} | Score: {bestScore:.3f}")

def apply_best_scaler(cols, sName):
    temp = dfDim[cols].replace([np.inf, -np.inf], np.nan).fillna(dfDim[cols].median())
    if sName == 'Quantile': return QuantileTransformer(output_distribution='uniform', random_state=42).fit_transform(temp)
    if sName == 'MinMax': return MinMaxScaler().fit_transform(temp)
    return StandardScaler().fit_transform(temp)

# PCA
pcaCols = bestResults['PCA']['prunedCols']
pcaScaled = apply_best_scaler(pcaCols, bestResults['PCA']['scalerName'])
pcaEmb = PCA(n_components=2, random_state=42).fit_transform(pcaScaled)
bestResults['PCA'].update({'emb': pcaEmb, 'scaled': pcaScaled})

# t-SNE
tsneCols = bestResults['TSNE']['prunedCols']
tsneScaled = apply_best_scaler(tsneCols, bestResults['TSNE']['scalerName'])
baseTsneEmb = TSNE(n_components=2, perplexity=min(30, len(tsneScaled)-1), random_state=42).fit_transform(tsneScaled)
bestResults['TSNE'].update({'emb': baseTsneEmb, 'scaled': tsneScaled})
maxPerp = min(50, len(dfDim) - 1)
for perp in [10, 30, min(50, maxPerp)]:
    try:
        tempEmb = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(tsneScaled)
        sc = silhouette_score(tempEmb, labels)
        if sc > bestResults['TSNE'].get('score', -1):
            bestResults['TSNE'].update({'emb': tempEmb, 'scaled': tsneScaled, 'score': sc, 'perp': perp})
    except: pass

# UMAP
umapCols = bestResults['UMAP']['prunedCols']
umapScaled = apply_best_scaler(umapCols, bestResults['UMAP']['scalerName'])
umapBaseEmb = umap.UMAP(n_components=2, random_state=42).fit_transform(umapScaled, y=umapLabels)
bestResults['UMAP'].update({'emb': umapBaseEmb, 'scaled': umapScaled})
for nNeighbors in [5, 15, 30]:
    for minDist in [0.01, 0.1, 0.5]:
        try:
            tempEmb = umap.UMAP(n_components=2, n_neighbors=nNeighbors, min_dist=minDist, random_state=42).fit_transform(umapScaled, y=labels)
            sc = silhouette_score(tempEmb, labels)
            if sc > bestResults['UMAP'].get('score', -1):
                bestResults['UMAP'].update({'emb': tempEmb, 'scaled': umapScaled, 'score': sc, 'nNeighbors': nNeighbors, 'minDist': minDist})
        except: pass

# 3. EXTRACT LOADINGS FOR HEATMAPS
pcaModel = PCA(n_components=2, random_state=42).fit(bestResults['PCA']['scaled'])
pcaLoadingsDf = pd.DataFrame(np.abs(pcaModel.components_[:2]), columns=bestResults['PCA']['prunedCols'], index=['Comp 1', 'Comp 2']).T.sort_values(by='Comp 1', ascending=False).head(8)

def get_manifold_importance(X, embId, cols):
    safeX = np.array(X).astype(float)
    miScore = mutual_info_regression(safeX, embId, random_state=42)
    if miScore.max() > 0:
        miScore = miScore / miScore.max()
    return pd.Series(miScore, index=cols)

tsneLoadingsDf = pd.DataFrame({
    'Comp 1': get_manifold_importance(bestResults['TSNE']['scaled'], bestResults['TSNE']['emb'][:, 0], bestResults['TSNE']['prunedCols']),
    'Comp 2': get_manifold_importance(bestResults['TSNE']['scaled'], bestResults['TSNE']['emb'][:, 1], bestResults['TSNE']['prunedCols'])
}).sort_values(by='Comp 1', ascending=False).head(8)

umapLoadingsDf = pd.DataFrame({
    'Comp 1': get_manifold_importance(bestResults['UMAP']['scaled'], bestResults['UMAP']['emb'][:, 0], bestResults['UMAP']['prunedCols']),
    'Comp 2': get_manifold_importance(bestResults['UMAP']['scaled'], bestResults['UMAP']['emb'][:, 1], bestResults['UMAP']['prunedCols'])
}).sort_values(by='Comp 1', ascending=False).head(8)

# ============================
# UNSUPERVISED BASELINE MODELS
# ============================
def get_unsup_score(cols):
    temp = dfDim[cols].replace([np.inf, -np.inf], np.nan).fillna(dfDim[cols].median())
    scaled = StandardScaler().fit_transform(temp)
    try:
        pca = PCA(n_components=2, random_state=42).fit(scaled)
        varScore = np.sum(pca.explained_variance_ratio_)
        loadings = np.abs(pca.components_[0]) + np.abs(pca.components_[1])
        return varScore, loadings, scaled
    except: return -1, None, None

uCols = list(baseCols)
bestUVar, bestULoadings, uScaled = get_unsup_score(uCols)

# PASS 1: Unsupervised Backward Elimination (Drop variables with lowest variance contribution)
improved = True
while improved and len(uCols) > 4:
    improved = False
    leastImportantIdx = np.argmin(bestULoadings)
    colToDrop = uCols[leastImportantIdx]
    testCols = [c for c in uCols if c != colToDrop]
    varSc, loadings, scaled = get_unsup_score(testCols)
    if varSc > bestUVar:
        bestUVar = varSc
        bestULoadings = loadings
        uScaled = scaled
        uCols.remove(colToDrop)
        improved = True

# PASS 2: Unsupervised Forward Selection
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
        varSc, loadings, scaled = get_unsup_score(testCols)
        if varSc > bestStepVar:
            bestStepVar = varSc
            colToAdd = col
            stepLoadings = loadings
            stepScaled = scaled
    if bestStepVar > bestUVar:
        bestUVar = bestStepVar
        bestULoadings = stepLoadings
        uScaled = stepScaled
        uCols.append(colToAdd)
        uDropped.remove(colToAdd)
        improved = True

print(f"[Unsupervised] Pruned to {len(uCols)} vars | 2D Explained Variance: {bestUVar:.3f}")

pcaUnsupModel = PCA(n_components=2, random_state=42).fit(uScaled)
pcaUnsupEmb = pcaUnsupModel.transform(uScaled)
tsneUnsupEmb = TSNE(n_components=2, perplexity=min(30, len(uScaled)-1), random_state=42).fit_transform(uScaled)
umapUnsupEmb = umap.UMAP(n_components=2, random_state=42).fit_transform(uScaled) # No y=labels here!
pcaUnsupScore = silhouette_score(pcaUnsupEmb, labels)
tsneUnsupScore = silhouette_score(tsneUnsupEmb, labels)
umapUnsupScore = silhouette_score(umapUnsupEmb, labels)
pcaUnsupLoadingsDf = pd.DataFrame(np.abs(pcaUnsupModel.components_[:2]), columns=uCols, index=['Comp 1', 'Comp 2']).T.sort_values(by='Comp 1', ascending=False).head(8)

tsneUnsupLoadingsDf = pd.DataFrame({
    'Comp 1': get_manifold_importance(uScaled, tsneUnsupEmb[:, 0], uCols),
    'Comp 2': get_manifold_importance(uScaled, tsneUnsupEmb[:, 1], uCols)
})

umapUnsupLoadingsDf = pd.DataFrame({
    'Comp 1': get_manifold_importance(uScaled, umapUnsupEmb[:, 0], uCols),
    'Comp 2': get_manifold_importance(uScaled, umapUnsupEmb[:, 1], uCols)
})

miThreshold = 0.01

def applyMiPruning(loadingsDf, baseCols, algoName, baseEmb, baseScore):
    # Identify features that exceed the threshold on both component
    aliveFeatures = loadingsDf[(loadingsDf['Comp 1'] > miThreshold) & (loadingsDf['Comp 2'] > miThreshold)].index.tolist()
    prunedCols = [c for c in baseCols if c in aliveFeatures]
    # If no features were dropped, or too few remain, return the original data
    if len(prunedCols) == len(baseCols) or len(prunedCols) < 2:
        return baseEmb, baseScore, loadingsDf, prunedCols
    print(f"[Unsupervised {algoName}] MI Pruning triggered: Dropped {len(baseCols) - len(prunedCols)} noise variables.")
    # Re-scale and Re-embed with the strictly noise-free feature set
    tempData = dfDim[prunedCols].replace([np.inf, -np.inf], np.nan).fillna(dfDim[prunedCols].median())
    cleanScaled = StandardScaler().fit_transform(tempData)
    try:
        if algoName == 't-SNE':
            newEmb = TSNE(n_components=2, perplexity=min(30, len(cleanScaled)-1), random_state=42).fit_transform(cleanScaled)
        elif algoName == 'UMAP':
            newEmb = umap.UMAP(n_components=2, random_state=42).fit_transform(cleanScaled)
        newScore = silhouette_score(newEmb, labels)
        # Recalculate Loadings for the updated heatmaps
        newLoadings = pd.DataFrame({
            'Comp 1': get_manifold_importance(cleanScaled, newEmb[:, 0], prunedCols),
            'Comp 2': get_manifold_importance(cleanScaled, newEmb[:, 1], prunedCols)
        }).sort_values(by='Comp 1', ascending=False)
        return newEmb, newScore, newLoadings, prunedCols
    except:
        return baseEmb, baseScore, loadingsDf, baseCols

# Execute the polish across all three manifolds
tsneUnsupEmb, tsneUnsupScore, tsneUnsupLoadingsDf, tsneUnsupCols = applyMiPruning(
    tsneUnsupLoadingsDf, uCols, 't-SNE', tsneUnsupEmb, tsneUnsupScore)

umapUnsupEmb, umapUnsupScore, umapUnsupLoadingsDf, umapUnsupCols = applyMiPruning(
    umapUnsupLoadingsDf, uCols, 'UMAP', umapUnsupEmb, umapUnsupScore)

tsneUnsupLoadingsDf = tsneUnsupLoadingsDf.sort_values(by='Comp 1', ascending=False).head(8)
umapUnsupLoadingsDf = umapUnsupLoadingsDf.sort_values(by='Comp 1', ascending=False).head(8)

# Overwrite the plotting assignments so the charts reflect the cleaned data
dfDim['uPCA_1'], dfDim['uPCA_2'] = pcaUnsupEmb[:, 0], pcaUnsupEmb[:, 1]
dfDim['uTSNE_1'], dfDim['uTSNE_2'] = tsneUnsupEmb[:, 0], tsneUnsupEmb[:, 1]
dfDim['uUMAP_1'], dfDim['uUMAP_2'] = umapUnsupEmb[:, 0], umapUnsupEmb[:, 1]

figDim, axesDim = plt.subplots(4, 3, figsize=(24, 28))

dfDim['PCA_1'], dfDim['PCA_2'] = bestResults['PCA']['emb'][:, 0], bestResults['PCA']['emb'][:, 1]
dfDim['TSNE_1'], dfDim['TSNE_2'] = bestResults['TSNE']['emb'][:, 0], bestResults['TSNE']['emb'][:, 1]
dfDim['UMAP_1'], dfDim['UMAP_2'] = bestResults['UMAP']['emb'][:, 0], bestResults['UMAP']['emb'][:, 1]
dfDim['uPCA_1'], dfDim['uPCA_2'] = pcaUnsupEmb[:, 0], pcaUnsupEmb[:, 1]
dfDim['uTSNE_1'], dfDim['uTSNE_2'] = tsneUnsupEmb[:, 0], tsneUnsupEmb[:, 1]
dfDim['uUMAP_1'], dfDim['uUMAP_2'] = umapUnsupEmb[:, 0], umapUnsupEmb[:, 1]
dfDim['Shape'] = dfDim['isSquare'].map({True: 'Square', False: 'Rectangular'})

# Row 1: Targeted/Supervised Scatter
sns.scatterplot(data=dfDim, x='PCA_1', y='PCA_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[0, 0], legend=False)
axesDim[0, 0].set_title(f"Targeted PCA ({len(bestResults['PCA']['prunedCols'])} vars)\nScore: {bestResults['PCA']['score']:.2f}", fontsize=14, fontweight='bold')
sns.scatterplot(data=dfDim, x='TSNE_1', y='TSNE_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[0, 1], legend=False)
axesDim[0, 1].set_title(f"Targeted t-SNE ({len(bestResults['TSNE']['prunedCols'])} vars)\nScore: {bestResults['TSNE']['score']:.2f}", fontsize=14, fontweight='bold')
sns.scatterplot(data=dfDim, x='UMAP_1', y='UMAP_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[0, 2], legend=False)
axesDim[0, 2].set_title(f"Fully Supervised UMAP ({len(bestResults['UMAP']['prunedCols'])} vars)\nScore: {bestResults['UMAP'].get('score', 0):.2f}", fontsize=14, fontweight='bold')

# Row 2: Targeted/Supervised Heatmap
sns.heatmap(pcaLoadingsDf, cmap='viridis', annot=True, fmt=".2f", ax=axesDim[1, 0])
axesDim[1, 0].set_title('Targeted PCA Absolute Loadings', fontsize=12, fontweight='bold')
axesDim[1, 0].set_yticklabels([t.get_text().replace(' ', '\n') for t in axesDim[1, 0].get_yticklabels()], rotation=0)
sns.heatmap(tsneLoadingsDf, cmap='viridis', annot=True, fmt=".2f", ax=axesDim[1, 1])
axesDim[1, 1].set_title('Targeted t-SNE MI Scoress', fontsize=12, fontweight='bold')
axesDim[1, 1].set_yticklabels([t.get_text().replace(' ', '\n') for t in axesDim[1, 1].get_yticklabels()], rotation=0)
sns.heatmap(umapLoadingsDf, cmap='viridis', annot=True, fmt=".2f", ax=axesDim[1, 2])
axesDim[1, 2].set_title('Supervised UMAP MI Scores', fontsize=12, fontweight='bold')
axesDim[1, 2].set_yticklabels([t.get_text().replace(' ', '\n') for t in axesDim[1, 2].get_yticklabels()], rotation=0)

# Row 3: Unsupervised Scatter
sns.scatterplot(data=dfDim, x='uPCA_1', y='uPCA_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[2, 0])
axesDim[2, 0].set_title(f"Unsupervised PCA ({len(uCols)} vars)\nVar (R²): {bestUVar:.2f} | Score: {pcaUnsupScore:.2f}", fontsize=14, fontweight='bold')
axesDim[2, 0].legend(loc='upper left', fontsize=10)
sns.scatterplot(data=dfDim, x='uTSNE_1', y='uTSNE_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[2, 1], legend=False)
axesDim[2, 1].set_title(f"Unsupervised t-SNE ({len(uCols)} vars)\nVar (R²): {bestUVar:.2f} | Score: {tsneUnsupScore:.2f}", fontsize=14, fontweight='bold')
sns.scatterplot(data=dfDim, x='uUMAP_1', y='uUMAP_2', hue='matrixGroup', style='Shape', markers={'Square': 's', 'Rectangular': 'o'}, s=100, alpha=0.8, ax=axesDim[2, 2], legend=False)
axesDim[2, 2].set_title(f"Unsupervised UMAP ({len(uCols)} vars)\nVar (R²): {bestUVar:.2f} | Score: {umapUnsupScore:.2f}", fontsize=14, fontweight='bold')


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

def tune_clustering(emb_data):
    bestScore = -1
    bestLabels = None
    bestName = "None"
    # Test KMeans
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(emb_data)
        score = silhouette_score(emb_data, kmeans.labels_)
        if score > bestScore:
            bestScore, bestLabels, bestName = score, kmeans.labels_, f"KMeans (k={k})"
    # Test DBSCAN (Scale first for consistent eps distance)
    scaled_emb = StandardScaler().fit_transform(emb_data)
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
    labels, algoName, score = tune_clustering(embData)
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

treeFeatures = ['Topological Entropy', 'RCM Bandwidth', 'Strictly Diagonally Dominant Row Fraction', 'isSquare']
binaryTargets = ['isSvdFailed', 'Rank Collapse', 'Positive Definite', 'isNonCholesky', 'isIrreducible']

masterTreeData = dfTransformed.copy()

condCol = pd.to_numeric(masterTreeData['Condition Number'], errors='coerce')
msvCol = pd.to_numeric(masterTreeData['Minimum Singular Value'], errors='coerce')
masterTreeData['isSvdFailed'] = (condCol.isna() | np.isinf(condCol) | (condCol >= 1e15) | msvCol.isna()).astype(int)
blockCol = pd.to_numeric(masterTreeData['Num Dmperm Blocks'], errors='coerce')
masterTreeData['isIrreducible'] = ((blockCol <= 1) | blockCol.isna()).astype(int)
rankCol = pd.to_numeric(masterTreeData['Full Numerical Rank?'], errors='coerce')
masterTreeData['Rank Collapse'] = ((rankCol == 0) | rankCol.isna()).astype(int)
cholCol = pd.to_numeric(masterTreeData['Cholesky Candidate'], errors='coerce')
masterTreeData['isNonCholesky'] = ((cholCol == 0) | cholCol.isna()).astype(int)

for col in treeFeatures:
    if col != 'isSquare':
        masterTreeData[col] = pd.to_numeric(masterTreeData[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(-1)

figTrees, axesTrees = plt.subplots(nrows=5, ncols=1, figsize=(30, 60))
figTrees.suptitle("Decision Trees: Topological Rules for Matrix Constraints", fontsize=24, fontweight='bold')
axesTrees = axesTrees.flatten()

paramGrid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [5, 10],
    'criterion': ['gini', 'entropy']
}

# Iterate, Optimize, and Plot each Target
for idx, targetCol in enumerate(binaryTargets):
    currAx = axesTrees[idx]
    targetData = masterTreeData.copy()
    if targetCol != 'isSvdFailed':
        targetData[targetCol] = pd.to_numeric(targetData[targetCol], errors='coerce')
        targetData = targetData.dropna(subset=[targetCol])
    if len(targetData) < 20:
        currAx.set_axis_off()
        currAx.set_title(f"Target: {targetCol}\n(Insufficient Data)", fontsize=16)
        continue
    xTree = targetData[treeFeatures]
    yTree = (targetData[targetCol] > 0).astype(int)
    # Ensure there are at least two classes to split
    if len(np.unique(yTree)) < 2:
        currAx.set_axis_off()
        currAx.set_title(f"Target: {targetCol}\n(Zero Target Variance)", fontsize=16, color='maroon')
        continue
    # Grid Search for the mathematically optimal tree structure
    baseTree = DecisionTreeClassifier(random_state=42)
    gridSearch = GridSearchCV(baseTree, paramGrid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
    gridSearch.fit(xTree, yTree)
    bestTree = gridSearch.best_estimator_
    if targetCol == 'isSvdFailed':
        classNames = ['SVD Success', 'SVD Failed']
    else:
        classNames = [f'Not {targetCol}', f'Is {targetCol}']
    plot_tree(bestTree, feature_names=treeFeatures, class_names=classNames, filled=True, rounded=True, ax=currAx, fontsize=10, proportion=False, impurity=False)
    currAx.set_title(f"Target: {targetCol}\nCV Balanced Acc: {gridSearch.best_score_:.3f} | Depth: {bestTree.max_depth}", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# ============================================
# COMPOSITE TREE CLUSTERING & MANIFOLD OVERLAY
# ============================================

treeFeatures = ['Topological Entropy', 'RCM Bandwidth', 'Strictly Diagonally Dominant Row Fraction', 'isSquare']
binaryTargets = ['isSvdFailed', 'Rank Collapse', 'Positive Definite', 'Cholesky Candidate']
evalData = dfDim.dropna(subset=treeFeatures + ['uUMAP_1', 'uUMAP_2', 'uTSNE_1', 'uTSNE_2', 'uPCA_1', 'uPCA_2']).copy()

condCol = pd.to_numeric(evalData['Condition Number'], errors='coerce')
msvCol = pd.to_numeric(evalData['Minimum Singular Value'], errors='coerce')
evalData['isSvdFailed'] = (condCol.isna() | np.isinf(condCol) | (condCol >= 1e15) | msvCol.isna()).astype(int)
blockCol = pd.to_numeric(evalData['Num Dmperm Blocks'], errors='coerce')
evalData['isIrreducible'] = ((blockCol <= 1) | blockCol.isna()).astype(int)
rankCol = pd.to_numeric(evalData['Full Numerical Rank?'], errors='coerce')
evalData['Rank Collapse'] = ((rankCol == 0) | rankCol.isna()).astype(int)
cholCol = pd.to_numeric(evalData['Cholesky Candidate'], errors='coerce')
evalData['isNonCholesky'] = ((cholCol == 0) | cholCol.isna()).astype(int)

for col in treeFeatures:
    if col != 'isSquare':
        evalData[col] = pd.to_numeric(evalData[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(-1)

evalData['isSquare'] = evalData['isSquare'].astype(int)
evalData['isSvdFailed'] = (evalData['Condition Number'].isna() | evalData['Minimum Singular Value'].isna()).astype(int)
trainedTrees = {}
# Train the 4 independent experts
paramGrid = {
    'max_depth': [3, 4],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [5, 10],
    'criterion': ['gini']
}

for targetCol in binaryTargets:
    targetData = evalData.copy()
    if targetCol != 'isSvdFailed':
        targetData[targetCol] = pd.to_numeric(targetData[targetCol], errors='coerce')
        targetData = targetData.dropna(subset=[targetCol])
    xTree = targetData[treeFeatures]
    yTree = (targetData[targetCol] > 0).astype(int)
    # If the target has no variance, skip training and store None
    if len(np.unique(yTree)) < 2:
        trainedTrees[targetCol] = None
        continue
    baseTree = DecisionTreeClassifier(random_state=42)
    gridSearch = GridSearchCV(baseTree, paramGrid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
    gridSearch.fit(xTree, yTree)
    trainedTrees[targetCol] = gridSearch.best_estimator_

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
    if row.get('isNonCholeskyPred', 0) == 1: flags.append('Non-Cholesky')
    if row.get('isIrreduciblePred', 0) == 1: flags.append('Irreducible')
    if len(flags) == 0: return 'Standard (No Flags)'
    return " + ".join(flags)

evalData['compositeProfile'] = evalData.apply(buildProfile, axis=1)

# Build the Composite State (The Deterministic Cluster)
constraintColors = {
    'isSvdFailedPred': 'magenta',
    'Rank CollapsePred': 'cyan',
    'Positive DefinitePred': 'red',
    'isNonCholeskyPred': 'yellow',
    'isIrreduciblePred': 'lime'
}

# Determine which matrices have NO flags
predCols = [c + 'Pred' for c in trainedTrees.keys() if c in trainedTrees and trainedTrees[c] is not None]
evalData['sumFlags'] = evalData[predCols].sum(axis=1)
      
condCol = pd.to_numeric(dfTransformed['Condition Number'], errors='coerce')
msvCol = pd.to_numeric(dfTransformed['Minimum Singular Value'], errors='coerce')
dfTransformed['isSvdFailed'] = (condCol.isna() | np.isinf(condCol) | (condCol >= 1e15) | msvCol.isna()).astype(int)
blockCol = pd.to_numeric(dfTransformed['Num Dmperm Blocks'], errors='coerce')
dfTransformed['isIrreducible'] = ((blockCol <= 1) | blockCol.isna()).astype(int)
rankCol = pd.to_numeric(dfTransformed['Full Numerical Rank?'], errors='coerce')
dfTransformed['Rank Collapse'] = ((rankCol == 0) | rankCol.isna()).astype(int)
cholCol = pd.to_numeric(dfTransformed['Cholesky Candidate'], errors='coerce')
dfTransformed['isNonCholesky'] = ((cholCol == 0) | cholCol.isna()).astype(int)

figComp, axesComp = plt.subplots(2, 3, figsize=(26, 16))
figComp.suptitle("Composite Decision Tree Predictions Overlaid on Topological Manifolds", fontsize=22, fontweight='bold')
manifolds = [('PCA', 'uPCA_1', 'uPCA_2'), ('t-SNE', 'uTSNE_1', 'uTSNE_2'), ('UMAP', 'uUMAP_1', 'uUMAP_2')]

for idx, (mName, xCol, yCol) in enumerate(manifolds):
    # TOP ROW: Alpha-Blended Constraint Predictions
    axTop = axesComp[0, idx]
    # Base Layer: Plot all matrices with NO constraints as faded grey
    baseMask = evalData['sumFlags'] == 0
    sns.scatterplot(data=evalData[baseMask], x=xCol, y=yCol, color='lightgrey', style='isSquare', markers={1: 's', 0: 'o'}, s=80, alpha=0.3, ax=axTop, label='Standard (No Flags)', legend=False)
    # Overplot Layer: Iterate through each constraint and plot it directly on top
    for targetCol in trainedTrees.keys():
        if trainedTrees[targetCol] is None: continue
        predCol = targetCol + 'Pred'
        mask = evalData[predCol] == 1
        if mask.sum() > 0:
            color = constraintColors.get(predCol, 'black')
            sns.scatterplot(data=evalData[mask], x=xCol, y=yCol, color=color, style='isSquare', markers={1: 's', 0: 'o'}, s=80, alpha=0.4, ax=axTop, label=targetCol, legend=False)
    axTop.set_title(f"Unsupervised {mName}\n(Alpha-Blended Constraints)", fontsize=14, fontweight='bold')
    axTop.set_xlabel(f"{mName} Dimension 1")
    axTop.set_ylabel(f"{mName} Dimension 2")
    axTop.grid(True, linestyle=':', alpha=0.5)
    if idx == 2:
        handles, labels = axTop.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axTop.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title="Constraint Layers")
    # BOTTOM ROW: Ground Truth SVD Failures
    axBot = axesComp[1, idx]
    # Context Layer: All successful matrices
    successMask = evalData['isSvdFailed'] == 0
    axBot.scatter(evalData.loc[successMask, xCol], evalData.loc[successMask, yCol], c='lightgrey', alpha=0.3, s=50, label='Successful SVD (Context)')
    # Failure Layer: The true SVD failures using their native, unaltered coordinates!
    failMask = evalData['isSvdFailed'] == 1
    failCount = failMask.sum()
    axBot.scatter(evalData.loc[failMask, xCol], evalData.loc[failMask, yCol], c='red', marker='*', s=200, edgecolor='black', linewidth=1, alpha=0.8, label='Failed SVD')
    axBot.set_title(f"Unsupervised {mName} Projection\n({failCount} True Failed Matrices)", fontsize=14, fontweight='bold')
    axBot.set_xlabel(f"{mName} Dimension 1")
    axBot.set_ylabel(f"{mName} Dimension 2")
    axBot.legend(loc='upper right', fontsize=11)
    axBot.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
print("\n--- CLUSTER VERIFICATION (Kruskal-Wallis) ---")
print(f"Features verified as distinct across clusters: {dfVerification['Statistically Distinct'].sum()} / {len(dfVerification)}")

# =========================================
# LOCAL EXPERTS: XGBOOST & SHAP PER CLUSTER
# =========================================
condCol = pd.to_numeric(dfTransformed['Condition Number'], errors='coerce')
msvCol = pd.to_numeric(dfTransformed['Minimum Singular Value'], errors='coerce')
dfTransformed['isSvdFailed'] = (condCol.isna() | np.isinf(condCol) | (condCol >= 1e15) | msvCol.isna()).astype(int)
blockCol = pd.to_numeric(dfTransformed['Num Dmperm Blocks'], errors='coerce')
dfTransformed['isIrreducible'] = ((blockCol <= 1) | blockCol.isna()).astype(int)
rankCol = pd.to_numeric(dfTransformed['Full Numerical Rank?'], errors='coerce')
dfTransformed['Rank Collapse'] = ((rankCol == 0) | rankCol.isna()).astype(int)
cholCol = pd.to_numeric(dfTransformed['Cholesky Candidate'], errors='coerce')
dfTransformed['isNonCholesky'] = ((cholCol == 0) | cholCol.isna()).astype(int)

expertTargets = ['Condition Number', 'Matrix Norm', 'Rank Collapse']

figShap, axesShap = plt.subplots(nrows=len(expertTargets), ncols=len(lockedProfiles), figsize=(26, 7 * len(expertTargets)))
figShap.suptitle("Local Expert Predictors: SHAP Impact by Topological Profile", fontsize=22, fontweight='bold')

for tIdx, targetCol in enumerate(expertTargets):
    for pIdx, profileName in enumerate(lockedProfiles):
        ax = axesShap[tIdx, pIdx]
        localDf = dfTransformed[dfTransformed['compositeProfile'] == profileName].copy()
        xgbData = localDf[[targetCol] + expertFeatures].copy()
        if targetCol in ['Condition Number', 'Matrix Norm']:
            xgbData[targetCol] = xgbData[targetCol].replace([np.inf, -np.inf], np.nan).fillna(1e35) # Impute NaN with extreme magnitude
        elif targetCol == 'Rank Collapse':
            xgbData[targetCol] = xgbData[targetCol].replace([np.inf, -np.inf], np.nan).fillna(1) # Impute NaN as a confirmed collapse
        if 'isInfinitePositivity' in xgbData.columns:
            xgbData['isInfinitePositivity'] = xgbData['isInfinitePositivity'].astype(int)
        if targetCol not in ['isInfinitePositivity']:
            xgbData[targetCol] = pd.to_numeric(xgbData[targetCol], errors='coerce').astype(float)
        for col in expertFeatures:
            if col != 'isInfinitePositivity':
                safeCol = pd.to_numeric(xgbData[col], errors='coerce').astype(float)
                xgbData[col] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
        # Cap any straggling algorithmic infinites to prevent XGBoost DMatrix crashes
        xgbData[expertFeatures] = xgbData[expertFeatures].replace([np.inf, -np.inf], np.nan)
        xgbData = xgbData.dropna(subset=[targetCol])
        if len(xgbData) < 15:
            ax.set_axis_off()
            ax.set_title(f"{profileName} | {targetCol}\n(Insufficient Data)", fontsize=11)
            continue
        xXgb = xgbData[expertFeatures]
        yXgb = xgbData[targetCol]
        isClassification = targetCol in ['Rank Collapse', 'Positive Definite', 'Cholesky Candidate', 'isSvdFailed', 'isIrreducible', 'isNonCholesky']
        if isClassification:
            yXgb = (yXgb > 0).astype(int)
            posCount = sum(yXgb)
            negCount = len(yXgb) - posCount
            if posCount == 0 or negCount == 0:
                ax.set_axis_off()
                ax.set_title(f"{profileName} | {targetCol}\n(Zero Target Variance)", fontsize=11, color='maroon')
                continue
            posWeight = negCount / posCount if posCount > 0 else 1
            model = xgb.XGBClassifier(n_estimators=75, max_depth=4, learning_rate=0.1, random_state=42, scale_pos_weight=posWeight)
        else:
            if targetCol in ['Condition Number', 'Matrix Norm']: yXgb = np.log1p(np.clip(yXgb, 0, 1e35))
            model = xgb.XGBRegressor(n_estimators=75, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(xXgb, yXgb)
        explainer = shap.TreeExplainer(model)
        shapValues = explainer.shap_values(xXgb)
        if isinstance(shapValues, list): shapValues = shapValues[1]
        plt.sca(ax)
        cleanFeatureNames = [c.replace(' ', '\n') for c in expertFeatures]
        shap.summary_plot(shapValues, xXgb, feature_names=cleanFeatureNames, max_display=6, show=False, plot_size=None)
        ax.set_title(f"{profileName}\nTarget: {targetCol}", fontsize=14, fontweight='bold')
ax.set_xlabel("SHAP Value (Impact)", fontsize=10)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

# ==============================================
# STATISTICAL VERIFICATION (HEATMAP & LINEARITY)
# ==============================================
matrixTargets = ['Condition Number', 'Matrix Norm', 'isSvdFailed', 'Rank Collapse', 'isIrreducible', 'isNonCholesky']
profileCounts = evalData['compositeProfile'].value_counts()
lockedProfiles = [p for p in profileCounts.index if p != 'Unknown' and profileCounts[p] >= 5]
finalResults = []

for tIdx, targetCol in enumerate(matrixTargets):
    for pIdx, profileName in enumerate(lockedProfiles):
        localDf = dfTransformed[dfTransformed['compositeProfile'] == profileName].copy()
        testData = localDf[[targetCol] + expertFeatures].copy()
        testData[targetCol] = pd.to_numeric(testData[targetCol], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if targetCol in ['Condition Number', 'Matrix Norm']:
            testData[targetCol] = testData[targetCol].fillna(1e35)
        elif targetCol in ['Rank Collapse', 'isSvdFailed']:
            testData[targetCol] = testData[targetCol].fillna(1)
        else:
            testData[targetCol] = testData[targetCol].fillna(0)
        testData[expertFeatures] = testData[expertFeatures].replace([np.inf, -np.inf], np.nan).fillna(0)
        testData = testData.dropna(subset=[targetCol])
        if len(testData) < 3:
            finalResults.append({'Target': targetCol, 'Profile': profileName, 'Top Predictor': "Insufficient Data", 'MI Score': 0.0, 'Linear PR-AUC': 0.0, 'Linear R²': 0.0})
            continue
        xStat = testData[expertFeatures].copy()
        yStat = testData[targetCol]
        isClassification = targetCol in ['Rank Collapse', 'Positive Definite', 'Cholesky Candidate', 'isSvdFailed', 'isIrreducible', 'isNonCholesky']
        for col in expertFeatures:
            safeCol = pd.to_numeric(xStat[col], errors='coerce').astype(float)
            xStat[col] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
        xStat = xStat.replace([np.inf, -np.inf], np.nan).fillna(0)
        if isClassification:
            binaryY = (yStat > 0).astype(int)
            if len(np.unique(binaryY)) < 2:
                finalResults.append({'Target': targetCol, 'Profile': profileName, 'Top Predictor': "100% Homogeneous\n(Zero Variance)", 'MI Score': 0.0, 'Linear PR-AUC': 0.0, 'Linear R²': 0.0})
                continue
        else:
            if yStat.nunique() < 2:
                finalResults.append({'Target': targetCol, 'Profile': profileName, 'Top Predictor': "100% Homogeneous\n(Zero Variance)", 'MI Score': 0.0, 'Linear PR-AUC': 0.0, 'Linear R²': 0.0})
                continue
        # Remove highly correlated features
        corrMatrix = xStat.corr(method='spearman').abs()
        upperTri = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(bool))
        toDrop = [column for column in upperTri.columns if any(upperTri[column] > 0.85)]
        xFiltered = xStat.drop(columns=toDrop)
        if xFiltered.shape[1] == 0:
            finalResults.append({'Target': targetCol, 'Profile': profileName, 'Top Predictor': "Features Collinear", 'MI Score': 0.0, 'Linear PR-AUC': 0.0, 'Linear R²': 0.0})
            continue
        scaledX = StandardScaler().fit_transform(xFiltered)
        linPrAuc = np.nan
        linR2 = np.nan
        if isClassification:
            miScores = mutual_info_classif(xFiltered, binaryY, random_state=42)
            linModel = LogisticRegression(penalty='l1', solver='saga', C=0.1, random_state=42, max_iter=5000, class_weight='balanced')
            try:
                linModel.fit(scaledX, binaryY)
                yPredProb = linModel.predict_proba(scaledX)[:, 1]
                linPrAuc = average_precision_score(binaryY, yPredProb)
            except: pass
        else:
            qt = QuantileTransformer(output_distribution='uniform', random_state=42)
            qtY = qt.fit_transform(yStat.values.reshape(-1, 1)).ravel()
            miScores = mutual_info_regression(xFiltered, qtY, random_state=42)
            linModel = Lasso(alpha=0.1, random_state=42)
            try:
                linModel.fit(scaledX, qtY)
                linR2 = linModel.score(scaledX, qtY)
            except: pass
        miSeries = pd.Series(miScores, index=xFiltered.columns).sort_values(ascending=False)
        finalResults.append({
            'Target': targetCol,
            'Profile': profileName,
            'Top Predictor': miSeries.index[0],
            'MI Score': miSeries.iloc[0],
            'Linear PR-AUC': linPrAuc,
            'Linear R²': linR2
        })
dfProof = pd.DataFrame(finalResults)
dfProof['Max Linear'] = dfProof[['Linear PR-AUC', 'Linear R²']].max(axis=1).fillna(0)
dfProof['Non-Linear Gap'] = dfProof['MI Score'] - dfProof['Max Linear']
dfGap = dfProof.dropna(subset=['Non-Linear Gap']).sort_values('Non-Linear Gap', ascending=False).head(15).copy().reset_index(drop=True)

def clean_label(row):
    pred_raw = str(row['Top Predictor']).split('\n')[0].split('Name:')[0]
    targ_raw = str(row['Target']).split('\n')[0].split('Name:')[0]
    # Regex: Eradicate anything inside brackets, and strip out stray index numbers
    pred_clean = re.sub(r'\(.*?\)', '', pred_raw)
    pred_clean = re.sub(r'^[0-9\s\.]+', '', pred_clean).strip()
    targ_clean = re.sub(r'\(.*?\)', '', targ_raw)
    targ_clean = re.sub(r'^[0-9\s\.]+', '', targ_clean).strip()
    # Replace the remaining spaces with newlines for the vertical Y-axis
    pred_final = pred_clean.replace(' ', '\n')
    return f"{pred_final}\n({targ_clean})"

dfGap['Gap Label'] = dfGap.apply(clean_label, axis=1)

# ========================================
# THE MASTER PROOF MATRIX
# ========================================
figMatrix, axMatrix = plt.subplots(figsize=(20, 10))
miPivot = dfProof.pivot(index='Target', columns='Profile', values='MI Score')
annotMatrix = dfProof.pivot(index='Target', columns='Profile', values='Top Predictor')
for col in miPivot.columns:
    for idx in miPivot.index:
        mask = (dfProof['Target'] == idx) & (dfProof['Profile'] == col)
        if mask.any():
            row = dfProof[mask].iloc[0]
            if "Homogeneous" in row['Top Predictor'] or "Insufficient" in row['Top Predictor'] or "Collinear" in row['Top Predictor']:
                annotMatrix.at[idx, col] = row['Top Predictor']
            else:
                lStr = f"{row['Max Linear']:.2f}" if pd.notna(row['Max Linear']) else "NaN"
                annotMatrix.at[idx, col] = f"{row['Top Predictor']}\nMI: {row['MI Score']:.2f} | Lin: {lStr}"
        else:
            annotMatrix.at[idx, col] = "N/A"
            miPivot.at[idx, col] = 0.0
sns.heatmap(miPivot, annot=annotMatrix, fmt="", cmap='YlGnBu', cbar_kws={'label': 'Mutual Information Score'}, linewidths=2, linecolor='white', ax=axMatrix, annot_kws={"size": 10, "weight": "bold"}, vmin=0)
axMatrix.set_title("Master Statistical Proof Matrix by Cluster Profile\n(Top Predictor, Mutual Info, & Linear Baseline)", fontsize=18, fontweight='bold')
axMatrix.set_xlabel("Topological Manifold Profiles (Clusters)", fontsize=14)
axMatrix.set_ylabel("Solver Constraints (Targets)", fontsize=14)
plt.tight_layout()
plt.show()

# ==============================
# LINEAR VS NON-LINEAR DOMINANCE
# ==============================
dfReg = dfProof[dfProof['Linear R²'].notna()].copy()
dfCls = dfProof[dfProof['Linear PR-AUC'].notna()].copy()

dfProof['Max Linear'] = dfProof[['Linear PR-AUC', 'Linear R²']].max(axis=1).fillna(0)
dfProof['Non-Linear Gap'] = dfProof['MI Score'] - dfProof['Max Linear']
dfGap = dfProof.dropna(subset=['Non-Linear Gap']).sort_values('Non-Linear Gap', ascending=False).head(15).copy()

cleanLabels = []
for rankIdx, rowDict in enumerate(dfGap.to_dict('records')):
    pred = str(rowDict['Top Predictor']).strip().replace('Product (', '\nProduct (').replace(' Ratio_is_missing', '\nRatio (Missing)').replace('n C', 'n\nC').replace('_is_inf', ' (Inf)')
    targ = str(rowDict['Target']).strip()
    cleanLabels.append(f"{pred}\n[{targ}]")
dfGap['Gap Label'] = cleanLabels

figProof = plt.figure(figsize=(26, 12))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)

# PANEL 1: Regression
axReg = figProof.add_subplot(gs[0])
if not dfReg.empty:
    sns.scatterplot(data=dfReg, x='Linear R²', y='MI Score', hue='Target', style='Profile', s=200, palette='tab10', ax=axReg)
    maxVal = max(dfReg['Linear R²'].max(), dfReg['MI Score'].max()) + 0.1
    axReg.plot([0, maxVal], [0, maxVal], 'k--', alpha=0.5, label='Linear Boundary')
    axReg.fill_between([0, maxVal], [0, maxVal], maxVal, color='blue', alpha=0.05)
    axReg.set_title("Regression: Linear vs Non-Linear Signal", fontsize=16, fontweight='bold')
    axReg.set_xlabel("Linear Baseline (Lasso R²)", fontsize=12)
    axReg.set_ylabel("Non-Linear Signal (MI Score)", fontsize=12)
    axReg.legend(loc='upper right', fontsize=10)
    axReg.grid(True, linestyle=':', alpha=0.6)

# PANEL 2: Classification
axCls = figProof.add_subplot(gs[1])
if not dfCls.empty:
    sns.scatterplot(data=dfCls, x='Linear PR-AUC', y='MI Score', hue='Target', style='Profile', s=200, palette='Set2', ax=axCls)
    maxVal = max(dfCls['Linear PR-AUC'].max(), dfCls['MI Score'].max()) + 0.1
    axCls.plot([0, maxVal], [0, maxVal], 'k--', alpha=0.5)
    axCls.fill_between([0, maxVal], [0, maxVal], maxVal, color='blue', alpha=0.05)
    axCls.set_title("Classification: Linear vs Non-Linear Signal", fontsize=16, fontweight='bold')
    axCls.set_xlabel("Linear Baseline (L1 Logistic PR-AUC)", fontsize=12)
    axCls.set_ylabel("Non-Linear Signal (MI Score)", fontsize=12)
    axCls.legend(loc='upper right', fontsize=10)
    axCls.grid(True, linestyle=':', alpha=0.6)

# PANEL 3: Non-Linearity Gap Bar Chart
axGap = figProof.add_subplot(gs[2])
if not dfGap.empty:
    sns.barplot(data=dfGap, x='Non-Linear Gap', y='Gap Label', hue='Target', dodge=False, palette='viridis', ax=axGap)
    axGap.set_title("Top Features Requiring Non-Linear Models", fontsize=16, fontweight='bold')
    axGap.set_xlabel("Non-Linearity Gap", fontsize=12)
    axGap.set_ylabel("")
    axGap.legend(loc='lower right', fontsize=10)
    axGap.tick_params(axis='y', labelsize=9)
    axGap.grid(True, axis='x', linestyle=':', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# =======================================================
# PARTIAL DEPENDENCE & ICE: GLOBAL RISK CURVES BY TARGET
# =======================================================

pdpTargets = ['Condition Number', 'Matrix Norm', 'isSvdFailed', 'Rank Collapse', 'isIrreducible', 'isNonCholesky']
numCols = 3
numRows = int(np.ceil(len(pdpTargets) / numCols))
figPDP, axesPDP = plt.subplots(nrows=numRows, ncols=numCols, figsize=(24, 8 * numRows))
figPDP.suptitle("Partial Dependence Risk Curves (Top Global Feature Impact)", fontsize=22, fontweight='bold')
axesPDP = axesPDP.flatten()

for tIdx, targetCol in enumerate(pdpTargets):
    ax = axesPDP[tIdx]
    pdpData = dfTransformed[[targetCol] + expertFeatures].copy()
    if targetCol in ['Condition Number', 'Matrix Norm']:
        pdpData[targetCol] = pd.to_numeric(pdpData[targetCol], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(1e35)
    elif targetCol == 'Rank Collapse':
        pdpData[targetCol] = pd.to_numeric(pdpData[targetCol], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(1)
    pdpData = pdpData.dropna(subset=[targetCol])
    if len(pdpData) < 20:
        ax.set_axis_off()
        continue
    xPdp = pdpData[expertFeatures].copy()
    for col in expertFeatures:
        safeCol = pd.to_numeric(xPdp[col], errors='coerce').astype(float)
        xPdp[col] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
    yPdp = pd.to_numeric(pdpData[targetCol], errors='coerce').astype(float)
    isClassification = targetCol in ['Rank Collapse', 'Positive Definite', 'Cholesky Candidate', 'isSvdFailed', 'isIrreducible', 'isNonCholesky']
    cleanX = xPdp.replace([np.inf, -np.inf], np.nan)
    cleanX = cleanX.fillna(cleanX.median()).fillna(0)
    if isClassification:
        binaryY = (yPdp > 0).astype(int)
        if len(np.unique(binaryY)) < 2:
            ax.set_axis_off()
            continue
        miScores = mutual_info_classif(cleanX, binaryY, random_state=42)
    else:
        qt = QuantileTransformer(output_distribution='uniform', random_state=42)
        qtY = qt.fit_transform(yPdp.values.reshape(-1, 1)).ravel()
        miScores = mutual_info_regression(cleanX, qtY, random_state=42)
    miSeries = pd.Series(miScores, index=cleanX.columns).sort_values(ascending=False)
    topFeature = miSeries.index[0]
    plt.sca(ax)
    try:
        if isClassification:
            posCount = sum(binaryY)
            posWeight = (len(binaryY) - posCount) / posCount if posCount > 0 else 1
            pdpModel = xgb.XGBClassifier(n_estimators=75, max_depth=4, random_state=42, scale_pos_weight=posWeight)
            pdpModel.fit(cleanX, binaryY)
        else:
            yReg = np.log1p(np.clip(yPdp, 0, 1e35))
            pdpModel = xgb.XGBRegressor(n_estimators=75, max_depth=4, random_state=42)
            pdpModel.fit(cleanX, yReg)
        PartialDependenceDisplay.from_estimator(
            pdpModel, cleanX, [topFeature],
            kind='both', ax=ax, ice_lines_kw={"color": "grey", "alpha": 0.1, "linewidth": 0.5},
            pd_line_kw={"color": "red", "linewidth": 3}
        )
        ax.set_title(f"Target: {targetCol}\nDominant Driver: {topFeature}", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
    except Exception:
        ax.set_title(f"Target: {targetCol}\n(PDP Render Failed)", fontsize=12)

for extraIdx in range(len(pdpTargets), len(axesPDP)):
    axesPDP[extraIdx].set_axis_off()

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.show()

# ====================================
# SHAP DEPENDENCE: ROOT CAUSE ANALYSIS
# ====================================
riskAnalysisTargets = ['isSvdFailed', 'Rank Collapse', 'Positive Definite', 'isIrreducible', 'isNonCholesky']

figRisk, axesRisk = plt.subplots(nrows=len(riskAnalysisTargets), ncols=2, figsize=(26, 8 * len(riskAnalysisTargets)))
figRisk.suptitle("Root Cause Analysis: Tree Gating Rules Linked to Local SHAP Interaction Thresholds", fontsize=24, fontweight='bold')

for tIdx, targetCol in enumerate(riskAnalysisTargets):
    axRules = axesRisk[tIdx, 0]
    axRules.set_axis_off()
    axShap = axesRisk[tIdx, 1]
    # Train Gating Tree Locally with balanced weights so it never predicts zero variance
    treeData = dfTransformed[treeFeatures + [targetCol]].copy()
    if targetCol == 'Rank Collapse':
        treeData[targetCol] = pd.to_numeric(treeData[targetCol], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(1)
    else:
        treeData[targetCol] = pd.to_numeric(treeData[targetCol], errors='coerce')
    treeData = treeData.dropna(subset=[targetCol])
    for col in treeFeatures:
        if col != 'isSquare':
            treeData[col] = pd.to_numeric(treeData[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(-1)
    xTreeData = treeData[treeFeatures]
    yTreeData = (treeData[targetCol] > 0).astype(int)
    gatingTree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    gatingTree.fit(xTreeData, yTreeData)
    treeRulesText = export_text(gatingTree, feature_names=treeFeatures)
    axRules.text(0.01, 0.95, f"--- {targetCol} Gating Rules ---\n(Probability of Risk State == 1)\n\n", ha='left', va='top', fontsize=14, fontweight='bold', color='black')
    axRules.text(0.01, 0.70, treeRulesText, ha='left', va='top', fontsize=11, fontfamily='monospace', color='black')
    highRiskMask = gatingTree.predict(xTreeData) == 1
    highRiskRegimeDf = treeData[highRiskMask].copy()
    highRiskRegimeDf = dfTransformed.loc[highRiskRegimeDf.index, [targetCol] + expertFeatures].copy()
    for col in expertFeatures:
        safeCol = pd.to_numeric(highRiskRegimeDf[col], errors='coerce').astype(float)
        highRiskRegimeDf[col] = np.sign(safeCol) * np.log1p(np.abs(safeCol))
    highRiskRegimeDf[expertFeatures] = highRiskRegimeDf[expertFeatures].replace([np.inf, -np.inf], np.nan)
    if len(highRiskRegimeDf) < 20 or len(np.unique((highRiskRegimeDf[targetCol] > 0).astype(int))) < 2:
        axShap.set_axis_off()
        axShap.text(0.5, 0.5, f"Insufficient variance in High-Risk Regime\nto train local expert for {targetCol}", ha='center', va='center', fontsize=12, color='maroon')
        continue
    xRisk = highRiskRegimeDf[expertFeatures]
    yRisk = (highRiskRegimeDf[targetCol] > 0).astype(int)
    posCount = sum(yRisk)
    posWeight = (len(yRisk) - posCount) / posCount if posCount > 0 else 1
    riskExpertModel = xgb.XGBClassifier(n_estimators=75, max_depth=4, learning_rate=0.1, random_state=42, scale_pos_weight=posWeight)
    riskExpertModel.fit(xRisk, yRisk)
    shapExplainer = shap.TreeExplainer(riskExpertModel)
    shapValues = shapExplainer.shap_values(xRisk)
    if isinstance(shapValues, list): shapValues = shapValues[1]
    localImportances = pd.Series(np.abs(shapValues).mean(axis=0), index=expertFeatures).sort_values(ascending=False)
    topExpertFeature = localImportances.index[0]
    topExpertFeatureIndex = expertFeatures.index(topExpertFeature)
    cleanFeatureNames = [c.replace(' ', '\n') for c in expertFeatures]
    plt.sca(axShap)
    try:
        xRiskVisual = xRisk.copy()
        for c in xRiskVisual.columns:
            xRiskVisual[c] = pd.to_numeric(xRiskVisual[c], errors='coerce').fillna(0)
            xData = xRiskVisual.iloc[:, topExpertFeatureIndex]
            if len(shapValues.shape) > 1:
                yShap = shapValues[:, topExpertFeatureIndex]
            else:
                yShap = shapValues
        axShap.scatter(xData, yShap, color='dodgerblue', alpha=0.7, s=80)
        axShap.set_title(f"{targetCol}: Local Thresholds for Expert Variable:\n{topExpertFeature}", fontsize=16, fontweight='bold')
        axShap.set_ylabel("SHAP Value (Risk Impact)")
        axShap.grid(True, linestyle=':', alpha=0.5)
    except:
        axShap.set_title(f"SHAP Dependence Render Failed for {topExpertFeature}", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# --------------------------------
# DEFINE PROGRESSIVE ROUTING GATES
# --------------------------------
colRenameDict = {'Strictly Diagonally Dominant Row Fraction': 'Diagonally Dominant\nRow Fraction'}
if 'Strictly Diagonally Dominant Row Fraction' in dfTransformed.columns:
    dfTransformed.rename(columns=colRenameDict, inplace=True)
if 'Strictly Diagonally Dominant Row Fraction' in evalData.columns:
    evalData.rename(columns=colRenameDict, inplace=True)

stage1Features = [
    'Diagonally Dominant\nRow Fraction', 'Directional Mean Bias', 
    'Brauer Max Product', 'Signed Frobenius Ratio', 'Numeric Symmetry'
]

stage2Features = [
    'Num Dmperm Blocks', 'RCM Bandwidth', 'Topological Entropy', 'Degeneracy Multiplier'
]

stage3Features = ['Fiedler Value']

if 'isDecomposable' not in dfTransformed.columns:
    dfTransformed['isDecomposable'] = (pd.to_numeric(dfTransformed['Num Dmperm Blocks'], errors='coerce').fillna(0) >= 5).astype(int)

if 'isDegenerate' not in dfTransformed.columns:
    dfTransformed['isDegenerate'] = (pd.to_numeric(dfTransformed['Degeneracy Multiplier'], errors='coerce').fillna(0) > 2.0).astype(int)

routingTargets = {
    'Rank Collapse': {'features': stage1Features, 'action': 'Route to Regularized Iterative (GMRES / LSQR)'},
    'isSvdFailed': {'features': stage1Features, 'action': 'Reject / Route to Arbitrary Precision Arithmetic'},
    'isNonCholesky': {'features': stage1Features, 'action': 'Route to Fast Cholesky Factorization', 'invertLogic': True},
    'isIrreducible': {'features': stage2Features, 'action': 'Route to Centralized Direct LU / MUMPS'},
    'isDecomposable': {'features': stage2Features, 'action': 'Route to Dantzig-Wolfe / Parallel ADMM'},
    'isDegenerate': {'features': stage2Features, 'action': 'Route to Exterior Point / Dual Simplex'}
}

# ---------------------------------------------------------
# TRAIN GATE THRESHOLDS
# ---------------------------------------------------------
learnedThresholds = {}
accuracyMetrics = []

for targetName, configDetails in routingTargets.items():
    allowedFeatures = configDetails['features']
    treeData = dfTransformed[allowedFeatures + [targetName]].copy()
    
    if targetName == 'Rank Collapse':
        treeData[targetName] = pd.to_numeric(treeData[targetName], errors='coerce').fillna(1)
    else:
        treeData[targetName] = pd.to_numeric(treeData[targetName], errors='coerce')
        
    treeData = treeData.dropna(subset=[targetName])
    
    for col in allowedFeatures:
        safeCol = pd.to_numeric(treeData[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(-1)
        treeData[col] = np.clip(safeCol, a_min=-1e35, a_max=1e35)
        
    xTree = treeData[allowedFeatures]
    yTree = (treeData[targetName] > 0).astype(int)
    
    if len(np.unique(yTree)) > 1:
        stumpModel = DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42)
        stumpModel.fit(xTree, yTree)
        yPred = stumpModel.predict(xTree)
        
        precisionVal = precision_score(yTree, yPred, zero_division=0)
        recallVal = recall_score(yTree, yPred, zero_division=0)
        f1Val = f1_score(yTree, yPred, zero_division=0)
        
        topFeatureIndex = stumpModel.tree_.feature[0]
        if topFeatureIndex != -2:
            topFeatureName = allowedFeatures[topFeatureIndex]
            thresholdValue = stumpModel.tree_.threshold[0]
            
            leftValueArray = stumpModel.tree_.value[1][0]
            rightValueArray = stumpModel.tree_.value[2][0]
            
            classIndices = np.where(stumpModel.classes_ == 1)[0]
            if len(classIndices) > 0:
                idx1 = classIndices[0]
                leftRisk = leftValueArray[idx1] / np.sum(leftValueArray)
                rightRisk = rightValueArray[idx1] / np.sum(rightValueArray)
            else:
                leftRisk = 0.0
                rightRisk = 0.0
                
            isGreaterThan = rightRisk > leftRisk
            learnedThresholds[targetName] = {
                'feature': topFeatureName,
                'threshold': thresholdValue,
                'isGreaterThan': isGreaterThan,
                'action': configDetails['action'],
                'invertLogic': configDetails.get('invertLogic', False)
            }
            accuracyMetrics.append({
                'Target Constraint': targetName,
                'Gating Feature': topFeatureName,
                'Precision': precisionVal,
                'Recall': recallVal,
                'F1 Score': f1Val
            })

# ----------------------
# EXECUTE AGENTIC ROUTER
# ----------------------
def agenticSolverRouter(row):
    # Stage 1: Algebraic Bipartite Scan
    ruleSvd = learnedThresholds.get('isSvdFailed')
    if ruleSvd:
        val = row.get(ruleSvd['feature'], -1)
        if (ruleSvd['isGreaterThan'] and val > ruleSvd['threshold']) or (not ruleSvd['isGreaterThan'] and val <= ruleSvd['threshold']):
            return ruleSvd['action'], 1
            
    ruleRank = learnedThresholds.get('Rank Collapse')
    if ruleRank:
        val = row.get(ruleRank['feature'], -1)
        if (ruleRank['isGreaterThan'] and val > ruleRank['threshold']) or (not ruleRank['isGreaterThan'] and val <= ruleRank['threshold']):
            return ruleRank['action'], 1
            
    ruleChol = learnedThresholds.get('isNonCholesky')
    if ruleChol:
        val = row.get(ruleChol['feature'], -1)
        hitRisk = (ruleChol['isGreaterThan'] and val > ruleChol['threshold']) or (not ruleChol['isGreaterThan'] and val <= ruleChol['threshold'])
        if not hitRisk:
            return ruleChol['action'], 1
            
    # Stage 2: Topological Graph Traversal
    ruleIrred = learnedThresholds.get('isIrreducible')
    if ruleIrred:
        val = row.get(ruleIrred['feature'], -1)
        if (ruleIrred['isGreaterThan'] and val > ruleIrred['threshold']) or (not ruleIrred['isGreaterThan'] and val <= ruleIrred['threshold']):
            return ruleIrred['action'], 2
            
    ruleDecomp = learnedThresholds.get('isDecomposable')
    if ruleDecomp:
        val = row.get(ruleDecomp['feature'], -1)
        if (ruleDecomp['isGreaterThan'] and val > ruleDecomp['threshold']) or (not ruleDecomp['isGreaterThan'] and val <= ruleDecomp['threshold']):
            return ruleDecomp['action'], 2
            
    ruleDegen = learnedThresholds.get('isDegenerate')
    if ruleDegen:
        val = row.get(ruleDegen['feature'], -1)
        if (ruleDegen['isGreaterThan'] and val > ruleDegen['threshold']) or (not ruleDegen['isGreaterThan'] and val <= ruleDegen['threshold']):
            return ruleDegen['action'], 2
            
    # Stage 3: Default / Spectral Fallback
    return 'Route to Primal-Dual Interior Point Method (IPM)', 3

evalData[['assignedSolver', 'computationStageRequired']] = evalData.apply(
    lambda r: pd.Series(agenticSolverRouter(r)), axis=1
)

# ==========================
# EMPIRICAL TIMING BENCHMARK
# ==========================

benchMat = sp.random(5000, 5000, density=0.001, format='csr', random_state=42)
benchMat = benchMat + benchMat.T 

# Stage 1
startStage1 = time.perf_counter()
rowSums = np.abs(benchMat).sum(axis=1).A1
diagonals = np.abs(benchMat.diagonal())
isDiagonallyDominant = (diagonals > (rowSums - diagonals)).sum()
frobeniusNorm = np.sqrt(benchMat.power(2).sum())
endStage1 = time.perf_counter()
stage1Ms = (endStage1 - startStage1) * 1000

# Stage 2
startStage2 = time.perf_counter()
rcmPermutation = csgraph.reverse_cuthill_mckee(benchMat)
numBlocks, blockLabels = csgraph.connected_components(benchMat)
endStage2 = time.perf_counter()
stage2Ms = (endStage2 - startStage2) * 1000

# Stage 3
startStage3 = time.perf_counter()
laplacianMat = sp.diags(rowSums) - benchMat
try:
    eigenValues, eigenVectors = splinalg.eigsh(laplacianMat, k=2, which='SM', tol=1e-2, maxiter=1000)
except Exception:
    pass
endStage3 = time.perf_counter()
stage3Ms = (endStage3 - startStage3) * 1000

dynamicTimingValues = [stage1Ms, stage2Ms, stage3Ms]
print(f"Benchmark Complete: Stage 1 = {stage1Ms:.2f}ms | Stage 2 = {stage2Ms:.2f}ms | Stage 3 = {stage3Ms:.2f}ms\n")

# ---------------------
# Rule Accuracy Heatmap
# ---------------------
dfAccuracy = pd.DataFrame(accuracyMetrics)
stageCounts = evalData['computationStageRequired'].value_counts().sort_index()

figPerf = plt.figure(figsize=(26, 18))
gsPerf = plt.GridSpec(2, 2, height_ratios=[1.2, 1], wspace=0.2, hspace=0.3)

axAcc = figPerf.add_subplot(gsPerf[0, 0])
if not dfAccuracy.empty:
    accPivot = dfAccuracy.set_index('Target Constraint')[['Precision', 'Recall', 'F1 Score']]
    annotAcc = dfAccuracy.set_index('Target Constraint')[['Gating Feature']]
    
    sns.heatmap(accPivot, annot=np.tile(annotAcc['Gating Feature'].values[:, None], (1, 3)), fmt="", cmap='viridis',
                cbar_kws={'label': 'Metric Score'}, linewidths=1, linecolor='white', ax=axAcc,
                annot_kws={"size": 11, "weight": "bold"}, vmin=0, vmax=1)
    axAcc.set_title("Single-Stump Rule Accuracy by Target\n(Gating Features)", fontsize=16, fontweight='bold')
    axAcc.set_ylabel("Solver Constraint", fontsize=14)
    axAcc.set_xlabel("Performance Metrics", fontsize=14)

# ----------------------------
# Agentic Routing Flow Diagram
# ----------------------------
axFlow = figPerf.add_subplot(gsPerf[0, 1])
axFlow.set_axis_off()
axFlow.set_title("Agentic Solver Decision Tree (Progressive Sub-Gates)", fontsize=16, fontweight='bold')

def getGateText(targetKey, fallbackName, invert=False):
    rule = learnedThresholds.get(targetKey)
    if rule:
        isGreater = rule['isGreaterThan']
        if invert: isGreater = not isGreater
        op = ">" if isGreater else "<="
        feat = rule['feature'].replace(' ', '\n')
        return f"IF {feat}\n{op} {rule['threshold']:.2f}"
    return f"IF {fallbackName}\nTriggered"

stageProps = dict(boxstyle="square,pad=0.6", fc="lightgrey", ec="black", lw=2)
gateProps = dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="darkorange", lw=2)
solvProps = dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="darkgreen", lw=2)
arrowProps = dict(facecolor='black', shrink=0.05, width=2, headwidth=8)
passProps = dict(facecolor='gray', shrink=0.05, width=2, headwidth=8, alpha=0.5)

axFlow.text(0.15, 0.95, "STAGE 1 COMPUTE\nO(NNZ) Algebraic", ha="center", va="center", bbox=stageProps, fontsize=12, fontweight='bold')
axFlow.text(0.15, 0.75, getGateText('isSvdFailed', 'SVD Risk'), ha="center", va="center", bbox=gateProps, fontsize=10, fontweight='bold')
axFlow.text(0.15, 0.55, getGateText('Rank Collapse', 'Rank Risk'), ha="center", va="center", bbox=gateProps, fontsize=10, fontweight='bold')
axFlow.text(0.15, 0.35, getGateText('isNonCholesky', 'Symmetry', invert=True), ha="center", va="center", bbox=gateProps, fontsize=10, fontweight='bold')

axFlow.text(0.40, 0.75, "Arbitrary Precision\n/ Reject", ha="center", va="center", bbox=solvProps, fontsize=11, fontweight='bold')
axFlow.text(0.40, 0.55, "Iterative Projection\n(GMRES)", ha="center", va="center", bbox=solvProps, fontsize=11, fontweight='bold')
axFlow.text(0.40, 0.35, "Fast Cholesky\nFactorization", ha="center", va="center", bbox=solvProps, fontsize=11, fontweight='bold')

axFlow.text(0.65, 0.95, "STAGE 2 COMPUTE\nO(N+NNZ) Traversal", ha="center", va="center", bbox=stageProps, fontsize=12, fontweight='bold')
axFlow.text(0.65, 0.75, getGateText('isIrreducible', 'Irreducible'), ha="center", va="center", bbox=gateProps, fontsize=10, fontweight='bold')
axFlow.text(0.65, 0.55, getGateText('isDecomposable', 'Decomposable'), ha="center", va="center", bbox=gateProps, fontsize=10, fontweight='bold')
axFlow.text(0.65, 0.35, getGateText('isDegenerate', 'Degenerate'), ha="center", va="center", bbox=gateProps, fontsize=10, fontweight='bold')

axFlow.text(0.90, 0.75, "Centralized Direct\n(MUMPS LU)", ha="center", va="center", bbox=solvProps, fontsize=11, fontweight='bold')
axFlow.text(0.90, 0.55, "Parallel Distributed\n(Dantzig-Wolfe)", ha="center", va="center", bbox=solvProps, fontsize=11, fontweight='bold')
axFlow.text(0.90, 0.35, "Exterior Point\n/ Dual Simplex", ha="center", va="center", bbox=solvProps, fontsize=11, fontweight='bold')

axFlow.text(0.65, 0.15, "STAGE 3 COMPUTE\nIterative Fallback", ha="center", va="center", bbox=stageProps, fontsize=12, fontweight='bold')
axFlow.text(0.90, 0.15, "Primal-Dual\nInterior Point", ha="center", va="center", bbox=solvProps, fontsize=11, fontweight='bold')

for yNode in [0.75, 0.55, 0.35]:
    axFlow.annotate("", xy=(0.28, yNode), xytext=(0.22, yNode), arrowprops=arrowProps)
    axFlow.annotate("", xy=(0.78, yNode), xytext=(0.72, yNode), arrowprops=arrowProps)

axFlow.annotate("", xy=(0.15, 0.82), xytext=(0.15, 0.88), arrowprops=passProps)
axFlow.annotate("", xy=(0.15, 0.62), xytext=(0.15, 0.68), arrowprops=passProps)
axFlow.annotate("", xy=(0.15, 0.42), xytext=(0.15, 0.48), arrowprops=passProps)
axFlow.annotate("Ambiguous", xy=(0.65, 0.98), xytext=(0.15, 0.28), arrowprops=dict(facecolor='gray', shrink=0.05, width=2, headwidth=8, alpha=0.3, connectionstyle="angle3,angleA=0,angleB=90"), ha='center')

axFlow.annotate("", xy=(0.65, 0.82), xytext=(0.65, 0.88), arrowprops=passProps)
axFlow.annotate("", xy=(0.65, 0.62), xytext=(0.65, 0.68), arrowprops=passProps)
axFlow.annotate("", xy=(0.65, 0.42), xytext=(0.65, 0.48), arrowprops=passProps)
axFlow.annotate("", xy=(0.65, 0.22), xytext=(0.65, 0.28), arrowprops=passProps)
axFlow.annotate("", xy=(0.78, 0.15), xytext=(0.72, 0.15), arrowprops=arrowProps)

axFlow.set_xlim(0, 1.0)
axFlow.set_ylim(0, 1.05)

# ------------------------
# Empirical Latency Timing
# ------------------------
axTime = figPerf.add_subplot(gsPerf[1, 0])
timingLabels = ['Stage 1 Compute\nO(NNZ) Algebraic', 'Stage 2 Compute\nO(N + NNZ) Traversal', 'Stage 3 Compute\nO(k * NNZ) Spectral']
timingValues = dynamicTimingValues

timeBars = axTime.bar(timingLabels, timingValues, color=['dodgerblue', 'mediumseagreen', 'crimson'], edgecolor='black', linewidth=1.5)
axTime.set_yscale('log')
axTime.set_title("Empirical Computational Latency per Stage (Log Scale)", fontsize=16, fontweight='bold')
axTime.set_ylabel("Execution Time (milliseconds)", fontsize=14)
axTime.grid(True, axis='y', which='both', linestyle=':', alpha=0.6)

maxTime = max(timingValues) if timingValues else 1
axTime.set_ylim(min(timingValues)*0.5, maxTime * 10)

for barObj in timeBars:
    height = barObj.get_height()
    axTime.text(barObj.get_x() + barObj.get_width()/2., height * 1.2,
                f"{height:.2f} ms", ha='center', va='bottom', fontsize=12, fontweight='bold')

# ----------------------
# Tier Resolution Funnel
# ----------------------
axStage = figPerf.add_subplot(gsPerf[1, 1])
totalMatrices = len(evalData)
s1Count = stageCounts.get(1, 0)
s2Count = stageCounts.get(2, 0)
s3Count = stageCounts.get(3, 0)

funnelLabels = ['Total Matrices\n(Input)', 'Resolved at Stage 1', 'Resolved at Stage 2', 'Routed to Stage 3\n(Fallback)']
funnelValues = [totalMatrices, s1Count, s2Count, s3Count]
funnelColors = ['lightgrey', 'dodgerblue', 'mediumseagreen', 'crimson']

bars = axStage.bar(funnelLabels, funnelValues, color=funnelColors, edgecolor='black', linewidth=1.5)
axStage.set_title("Computational Tier Resolution Funnel", fontsize=16, fontweight='bold')
axStage.set_ylabel("Number of Matrices", fontsize=14)
axStage.grid(True, axis='y', linestyle=':', alpha=0.6)

maxFunnel = max(funnelValues) if funnelValues else 1
axStage.set_ylim(0, maxFunnel * 1.3)

for idx, barObj in enumerate(bars):
    height = barObj.get_height()
    percentage = (height / totalMatrices) * 100 if totalMatrices > 0 else 0
    axStage.text(barObj.get_x() + barObj.get_width()/2., height + (maxFunnel*0.02),
                 f"{height}\n({percentage:.1f}%)", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nFinal Recommended Solver Distribution:")
print(evalData['assignedSolver'].value_counts().to_string())
