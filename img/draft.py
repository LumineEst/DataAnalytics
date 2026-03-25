# Laplacian Exploratory Analysis (Joel)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ast
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Reading in the Data and forming a merged dataset
laplacianDf = pd.read_csv('laplacian.csv')
rectDf = pd.read_csv('matrixdata.csv')

mergedDf = pd.merge(rectDf, laplacianDf, on='Name', suffixes=(' (Original)', ' (Laplacian)'))
mergedDf['isRectangular'] = mergedDf['Num Rows (Original)'] != mergedDf['Num Cols (Original)']
analysisDf = mergedDf[mergedDf['isRectangular']].copy()

# Function to calculate the minimum and maximum values, as well as their span, within an input
def extract_bounds(val):
    if pd.isna(val): return np.nan, np.nan, np.nan
    try:
        intervals = ast.literal_eval(val)
        if not intervals: return np.nan, np.nan, np.nan
        minVal = min([interval[0] for interval in intervals])
        maxVal = max([interval[1] for interval in intervals])
        return minVal, maxVal, maxVal - minVal
    except:
        return np.nan, np.nan, np.nan

analysisDf['Gershgorin Min'], analysisDf['Gershgorin Max'], analysisDf['Gershgorin Span'] = zip(*analysisDf['Gershgorin Discs (Laplacian)'].apply(extract_bounds))
analysisDf['Density (Original)'] = analysisDf['Nonzeros (Original)'] / (analysisDf['Num Rows (Original)'] * analysisDf['Num Cols (Original)'])
lapNodes = analysisDf['Num Rows (Original)'] + analysisDf['Num Cols (Original)']
lapNonZeros = lapNodes + (2 * analysisDf['Nonzeros (Original)'])
analysisDf['Density (Laplacian)'] = lapNonZeros / (lapNodes ** 2)
analysisDf['Brauer Max Product Root'] = np.sqrt(analysisDf['Brauer Max Product (Laplacian)'].clip(lower=0))
analysisDf['RCM Compression Ratio'] = analysisDf['RCM Bandwidth (Laplacian)'] / lapNodes

# The Row Fraction is an isolation metric in Laplacians
if 'Strictly Diagonally Dominant Row Fraction (Laplacian)' in analysisDf.columns:
    analysisDf['Isolated Node Fraction'] = analysisDf['Strictly Diagonally Dominant Row Fraction (Laplacian)']

# Rank Test and Cross Correlation
laplacianMetrics = ['Fiedler Value (Laplacian)', 'RCM Compression Ratio', 'Gershgorin Span', 'Isolated Node Fraction']
structuralPoints = ['Nonzeros (Original)', 'Structural Rank (Original)', 'Condition Number (Original)', 'Matrix Norm (Original)', 'Num Dmperm Blocks (Original)'] 
rankResults = []
for lap in laplacianMetrics:
    for sp in structuralPoints:
        valid = analysisDf[[lap, sp]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) > 2:
            corr, pval = spearmanr(valid[lap], valid[sp])
            rankResults.append({'Laplacian_Metric': lap.replace(' (Laplacian)', ''), 'Original_Point': sp.replace(' (Original)', ''), 'Spearman_rho': corr, 'P_value': pval})
corrData = analysisDf[laplacianMetrics + structuralPoints]. replace([np.inf, -np.inf], np.nan).dropna()
crossCorr = corrData.corr(method='spearman').loc[structuralPoints, laplacianMetrics]
crossCorr.index = [idx.replace(' (Original)', '').replace('Condition', 'Cond').replace('Structural', 'Struct').replace('Num Dmperm Blocks', 'DmPerm\nBlocks') for idx in crossCorr.index]
crossCorr.columns = [col_name.replace(' (Laplacian)', '').replace(' Span', '\nSpan').replace('RCM ', 'RCM\n') for col_name in crossCorr.columns]
crossCorr.columns = [col_name.replace('Strictly Diagonally Dominant Row Fraction', 'Diagonally\nDominant\nRow Fraction') for col_name in crossCorr.columns]

analysisDf['Min Dimension (Original)'] = analysisDf[['Num Rows (Original)', 'Num Cols (Original)']].min(axis=1)
analysisDf['Rank Ratio (Original)'] = analysisDf['Structural Rank (Original)'] / analysisDf['Min Dimension (Original)']
analysisDf['Is Disconnected'] = analysisDf['Num Dmperm Blocks (Original)'] > 1
analysisDf['Rank Deficient (Original)'] = np.where(analysisDf['sprank(A)-rank(A) (Original)'] > 0, 'Rank Deficient', 'Full Rank')
analysisDf['Bound Looseness Ratio'] = (analysisDf['Gershgorin Max'] / analysisDf['Brauer Max Product Root'].replace(0, np.nan)).clip(lower=1.0)

# PCA Manifold Projection
manifoldFeatures = [
    # Topological
    'RCM Compression Ratio', 'Fiedler Value (Laplacian)', 'Brauer Max Center Distance (Laplacian)', 'Isolated Node Fraction',
    # Algebraic
    'Directional Mean Bias (Laplacian)', 'Signed Frobenius Ratio (Laplacian)', 'Brauer Max Product Root', 'Gershgorin Span', 
]

# PCA 1 and 2 Loadings
manifoldDf = analysisDf.replace([np.inf, -np.inf], np.nan).dropna(subset=manifoldFeatures + ['Kind (Original)']).copy()
xScaled = StandardScaler().fit_transform(manifoldDf[manifoldFeatures])
pca = PCA(n_components=2)
topKinds = manifoldDf['Kind (Original)'].value_counts().nlargest(5).index
manifoldDf['Kind_Grouped'] = manifoldDf['Kind (Original)'].apply(lambda x: x if x in topKinds else 'Other')
manifoldDf['Is_LP'] = manifoldDf['Kind_Grouped'] == 'Linear Programming Problem'
manifoldDf = manifoldDf.sort_values(by='Is_LP', ascending=False)
manifoldDf[['PCA1', 'PCA2']] = pca.fit_transform(xScaled)
manifoldFeatures = [col.replace(' (Laplacian)', '') for col in manifoldFeatures]
manifoldFeatures = [col.replace('Directional Mean Bias', 'Directional\nMean Bias') for col in manifoldFeatures]
manifoldFeatures = [col.replace('Brauer Max Product Root', 'Brauer Max\nProduct Root') for col in manifoldFeatures]
manifoldFeatures = [col.replace('RCM Compression Ratio', 'RCM Compression\nRatio') for col in manifoldFeatures]
pcaLoadings = pd.DataFrame(pca.components_, columns=manifoldFeatures, index=['PC 1', 'PC 2'])

# Predictive Random Forest Model
mlFeatures = [
    'Isolated Node Fraction', 'Directional Mean Bias (Laplacian)', 'Gershgorin Span',
    'Brauer Mean Product (Laplacian)', 'Fiedler Value (Laplacian)', 'RCM Compression Ratio'
]
analysisDf['Rank Deficient (Original)'] = np.where(analysisDf['sprank(A)-rank(A) (Original)'] > 0, 'Rank Deficient', 'Full Rank')
mlDf = analysisDf[mlFeatures + ['Rank Deficient (Original)']].copy()
mlDf = mlDf[mlFeatures].replace([np.inf, -np.inf], np.nan).dropna(subset=mlFeatures)

if len(mlDf['Rank Deficient (Original)'].unique()) > 1:
    xTrain, xTest, yTrain, yTest = train_test_split(mlDf[mlFeatures], mlDf['Rank Deficient (Original)'], test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(xTrain, yTrain)
    acc = accuracy_score(yTest, rf.predict(xTest))
    importances = pd.Series(rf.feature_importances_, index=mlFeatures).sort_values(ascending=False)

# Creating Matrix Type Color-Map
top5Kinds = analysisDf['Kind (Original)'].value_counts().nlargest(5).index
analysisDf['Kind_Grouped'] = analysisDf['Kind (Original)'].apply(lambda x: x if x in top5Kinds else 'Other')
analysisDf['Is_LP'] = analysisDf['Kind_Grouped'] == 'Linear Programming Problem'
analysisDf = analysisDf.sort_values(by='Is_LP', ascending=False)
kinds = ['Linear Programming Problem'] + [k for k in top5Kinds if k != 'Linear Programming Problem'] + ['Other']
kindHues = dict(zip(kinds, sns.color_palette('Set2', n_colors=len(kinds))))

figA, axesA = plt.subplots(2, 3, figsize=(24, 16))

# Plot 1: Rank Preservation
sns.histplot(data=analysisDf, x='Rank Ratio (Original)', hue='Kind_Grouped', legend=False,
             multiple='stack', bins=20, palette=kindHues, hue_order=kinds[::-1], ax=axesA[0,0]
)
axesA[0,0].axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.8)
axesA[0,0].set_title("Original Structural Rank Preservation", fontweight='bold')
axesA[0,0].set_xlabel('Structural Rank / Minimum Dimension')
custHandles = [mpatches.Patch(color=kindHues[kind], label=kind) for kind in kinds]
axesA[0,0].legend(handles=custHandles, title='Matrix Kind', loc='upper left', fontsize=9, title_fontsize=10)

# Plot 2: Spearman Heatmap
sns.heatmap(crossCorr, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1, ax=axesA[0,1])
axesA[0,1].set_title("Spearman Rank Correlation", fontweight='bold')

# Plot 3: PCA Taxonomy
topKinds = manifoldDf['Kind (Original)'].value_counts().nlargest(5).index
plot1Df = manifoldDf[manifoldDf['Kind (Original)'].isin(topKinds)].copy()
plot1Df = plot1Df.replace([np.inf, -np.inf], np.nan).dropna(subset-['Condition Number (Original)'])
plot1Df['Is_LP'] = plot1Df['Kind (Original)'] == 'Linear Programming Problem'
plot1Df = plot1Df.sort_values(by='Is_LP', ascending=False)
sns.scatterplot(data=plot1Df, x='PCA1', y='PCA2', hue='Kind_Grouped', palette=kindHues,
                size='Condition Number (Original)', sizes=(40, 400), size_norm=mcolors.LogNorm(), alpha=0.8, ax=axesA[0,2])
axesA[0,2].set_title('High-Dimensional Manifold Taxonomy (PCA)', fontweight='bold')

# Plot 4: PCA Loadings Heatmap 
sns.heatmap(pcaLoadings, annot=True, cmap='RdBu_r', center=0, ax=axesA[1,0], fmt=".2f", cbar_kws={'label': 'Feature Weight / Loading'})
axesA[1,0].set(xlabel='Proxy Metric', ylabel='Principle Components')
axesA[1,0].set_title('PCA Feature Loadings: PC1 & PC2', fontweight='bold')

# Plot 5: RF Feature Importance
sns.barplot(x=importances.values, y=[i.replace(' (Laplacian)', '')
    .replace('Brauer Mean Product', 'Brauer Mean\nProduct')
    .replace('Directional Mean Bias', 'Directional\nMean Bias') for i in importances.index], ax=axesA[1,1], palette='viridis')
axesA[1,1].set_title(f'RF Predictor: Original Rank Deficiency (Acc: {acc:.2f})', fontweight='bold')

# Plot 6: Dual Proxy Decision Space
decisionData = analysisDf.dropna(subset=['Isolated Node Fraction', 'Gershgorin Span', 'Rank Deficient (Original)']).copy()
sns.kdeplot(
    data=decisionData, x='Isolated Node Fraction', y='Gershgorin Span', 
    hue='Rank Deficient (Original)', fill=True, alpha=0.4, 
    palette='Set1', thresh=0.05, ax=axesA[1,2]
)
sns.scatterplot(
    data=decisionData, x='Isolated Node Fraction', y='Gershgorin Span', 
    hue='Rank Deficient (Original)', palette='Set1', alpha=0.8, legend=True, 
    ax=axesA[1,2], s=40, edgecolor='white', linewidth=0.5
)
axesA[1,2].set(yscale='log')
axesA[1,2].set_title('Dual-Proxy Decision Space (Top ML Features)', fontweight='bold', fontsize=14)
axesA[1,2].set_xlabel('Isolated Node Fraction (Topological)')
axesA[1,2].set_ylabel('Gershgorin Span (Algebraic) [Log]')
sns.move_legend(axesA[1,2], "upper right", title='Original Rank Status', fontsize=10)

plt.tight_layout()
plt.show()

figL, axesL = plt.subplots(3, 3, figsize=(24, 20))

# Plot 1: Density Shift
sns.scatterplot(
    data=analysisDf, y='Density (Laplacian)', x='Density (Original)', 
    ax=axesL[0,0], alpha=0.7, hue='Kind_Grouped', palette=kindHues, s=80
)
axesL[0,0].set(xscale='log', yscale='log', ylabel='Laplacian Density', xlabel='Original Density')
axesL[0,0].plot([1e-6, 1], [1e-6, 1], 'r--', alpha=0.5, label='1:1 Shift')
axesL[0,0].set_xlim(left=5e-5)
axesL[0,0].set_ylim(bottom=1e-5)
axesL[0,0].set_title("Structural Density Shift", fontweight='bold', fontsize=13)
axesL[0,0].legend(handles=custHandles, title='Matrix Kind', loc='upper left', fontsize=9)

# Plot 2: Solver Fill-In Routing
sns.scatterplot(
    data=analysisDf, x='RCM Compression Ratio', y='Nonzeros (Original)', 
    hue='Density (Original)', palette='magma', ax=axesL[0,1], alpha=0.7, s=80
)
axesL[0,1].set(xscale='log', yscale='log', xlabel='RCM Compression Ratio (Bandwidth / Nodes)')
axesL[0,1].set_title("Graph Compression vs Memory Footprint", fontweight='bold', fontsize=13)

# Plot 3: Bandwidth Collapse
blockData = analysisDf.replace([np.inf, -np.inf], np.nan).dropna(subset=['RCM Compression Ratio', 'Num Dmperm Blocks (Original)', 'Isolated Node Fraction'])
fitData = blockData[(blockData['Num Dmperm Blocks (Original)'] > 0) & (blockData['RCM Compression Ratio'] > 0)]
if not fitData.empty:
    xData, yData = fitData['RCM Compression Ratio'], fitData['Num Dmperm Blocks (Original)']
    m, c = np.polyfit(np.log10(xData), np.log10(yData), 1)
    xLine = np.logspace(np.log10(xData.min()), np.log10(xData.max()), 100)
    yLine = (10**c) * (xLine**m)
    axesL[0,2].plot(xLine, yLine, color='red', linestyle='--', linewidth=2, alpha=0.8)
sns.scatterplot(
    data=blockData, x='RCM Compression Ratio', y='Num Dmperm Blocks (Original)',
    hue='Isolated Node Fraction', palette='viridis_r', s=80, alpha=0.8, ax=axesL[0,2]
)
axesL[0,2].set(xscale='log', yscale='log', xlabel='RCM Compression Ratio')
axesL[0,2].set_title("Bandwidth Collapse vs. Structural Fracturing", fontweight='bold', fontsize=13)
sns.move_legend(axesL[0,2], "upper right", title='Isolated Node\nFraction', fontsize=9)

# Plot 4: Partitioning Efficiency
sns.scatterplot(
    data=analysisDf, x='Fiedler Value (Laplacian)', y='Num Dmperm Blocks (Original)',
    hue='Strongly Connect Components (Original)', size='Strongly Connect Components (Original)',
    size_norm=mcolors.LogNorm(), hue_norm=mcolors.LogNorm(), sizes=(20, 250), 
    palette='flare', alpha=0.7, ax=axesL[1,0]
)
axesL[1,0].set(xscale='symlog', yscale='log', linthresh=1e-5)
axesL[1,0].set_title("Fiedler Value vs Block Decomposition", fontweight='bold', fontsize=13)
axesL[1,0].set_xlim(left=-1e-5)
axesL[1,0].legend(title='Strongly Connect\nComponents', loc='upper right', fontsize=9)

# Plot 5: Graph Fracturing
nullityData = analysisDf.dropna(subset=['Null Space Dimension (Laplacian)', 'sprank(A)-rank(A) (Original)', 'Isolated Node Fraction'])
sns.scatterplot(data=nullityData, x='Null Space Dimension (Laplacian)', y='sprank(A)-rank(A) (Original)', hue='Isolated Node Fraction', palette='viridis_r', ax=axesL[1,1], alpha=0.8, s=80)
axesL[1,1].set(xscale='log', yscale='symlog', ylabel='Original Rank Defect (sprank - rank)')
axesL[1,1].set_title("Graph Nullity vs Algebraic Rank Defect", fontweight='bold', fontsize=13)
axesL[1,1].set_ylim(bottom=-0.5)

# Plot 6: Graph Irregularity
validNorm = analysisDf.dropna(subset=['Brauer Max Center Distance (Laplacian)', 'Matrix Norm (Original)', 'Rank Deficient (Original)'])
sns.kdeplot(data=validNorm, x='Brauer Max Center Distance (Laplacian)', y='Matrix Norm (Original)', hue='Rank Deficient (Original)', fill=True, alpha=0.3, palette='Set1', thresh=0.05, ax=axesL[1,2])
sns.scatterplot(data=validNorm, x='Brauer Max Center Distance (Laplacian)', y='Matrix Norm (Original)', hue='Rank Deficient (Original)', palette='Set1', ax=axesL[1,2], alpha=0.8, s=60)
axesL[1,2].set(xscale='log', yscale='log')
axesL[1,2].set(xlabel='Brauer Max Center Distance (Graph Irregularity)')
axesL[1,2].set_title("Graph Irregularity vs Matrix Norm", fontweight='bold', fontsize=13)

# Plot 7: The Degeneracy Predictor
degData = analysisDf.replace([np.inf, -np.inf], np.nan).dropna(subset=['Isolated Node Fraction', 'Rank Ratio (Original)'])
sns.scatterplot(
    data=degData, x='Isolated Node Fraction', y='Rank Ratio (Original)',
    hue='Kind_Grouped', palette=kindHues, s=80, alpha=0.7, ax=axesL[2,0]
)
axesL[2,0].set_title("Degeneracy Predictor: Dead Nodes vs. Rank Ratio", fontweight='bold', fontsize=13)
axesL[2,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
axesL[2,0].set_ylim(top=1.05, bottom=0.7)
axesL[2,0].legend(handles=custHandles, title='Matrix Kind', loc='lower left', fontsize=9)

# Plot 8: Dead Weight Distribution
validFrac = analysisDf[analysisDf['Isolated Node Fraction'] > 0]
sns.histplot(
    data=validFrac, x='Isolated Node Fraction', hue='Kind_Grouped', legend=False, 
    multiple='stack', bins=20, ax=axesL[2,1], palette=kindHues, hue_order=kinds[::-1]
)
axesL[2,1].set_title("Dead Weight Distribution by Matrix Kind", fontweight='bold', fontsize=13)
axesL[2,1].legend(handles=custHandles, title='Matrix Kind', loc='upper right', fontsize=9)

# Plot 9: Bivariate Fracturing KDE (NEW PLOT)
zagrebData = analysisDf.dropna(subset=['Brauer Mean Product (Laplacian)', 'Num Dmperm Blocks (Original)', 'Rank Deficient (Original)'])
sns.kdeplot(
    data=zagrebData, x='Brauer Mean Product (Laplacian)', y='Num Dmperm Blocks (Original)', 
    common_norm=False, hue='Rank Deficient (Original)', fill=True, alpha=0.3, 
    palette='Set1', thresh=0.05, ax=axesL[2,2]
)
sns.scatterplot(
    data=zagrebData, x='Brauer Mean Product (Laplacian)', y='Num Dmperm Blocks (Original)',
    hue='Rank Deficient (Original)', palette='Set1', alpha=0.8, legend=True, ax=axesL[2,2], s=40
)
axesL[2,2].set(xscale='log', yscale='log')
axesL[2,2].set_xlabel('Brauer Mean Product')
axesL[2,2].set_title("Branching Complexity vs Dmperm Fracturing", fontweight='bold', fontsize=13)
sns.move_legend(axesL[2,2], "upper right", title='Rank Status')


plt.tight_layout()
plt.show()

figM, axesM = plt.subplots(3, 3, figsize=(24, 20))

# Plot 1: Lipschitz Proxy
normData = analysisDf.dropna(subset=['Gershgorin Max', 'Matrix Norm (Original)'])
sns.scatterplot(
    data=normData, x='Gershgorin Max', y='Matrix Norm (Original)',
    hue='Kind_Grouped', palette=kindHues, s=80, alpha=0.7, ax=axesM[0,0]
)
axesM[0,0].set(xscale='log', yscale='log', xlabel='Gershgorin Max Bound (Algebraic)', ylabel='Original Matrix Norm (Max SVD)')
axesM[0,0].plot([1, 1e10], [1, 1e10], 'r--', alpha=0.5, label='1:1 Bound')
axesM[0,0].set_xlim(right=1e6)
axesM[0,0].set_ylim(top=1e6)
axesM[0,0].set_title("Lipschitz Proxy: Gershgorin Max vs Original Norm", fontweight='bold', fontsize=13)
axesM[0,0].legend(handles=custHandles, title='Matrix Kind', loc='upper left', fontsize=9)

# Plot 2: The Singularity Ceiling
sns.scatterplot(
    data=normData, x='Brauer Max Product Root', y='Matrix Norm (Original)', 
    hue='Kind_Grouped', palette=kindHues, s=80, alpha=0.7, ax=axesM[0,1]
)
axesM[0,1].set(xscale='log', yscale='log', xlabel='sqrt(Brauer Max Product)', ylabel='Original Matrix Norm (Max SVD)')
axesM[0,1].plot([1, 1e10], [1, 1e10], 'r--', alpha=0.5, label='1:1 Bound')
axesM[0,1].set_xlim(right=1e6)
axesM[0,1].set_ylim(top=1e6)
axesM[0,1].set_title("Singularity Ceiling: Brauer Bound vs Max SVD", fontweight='bold', fontsize=13)
axesM[0,1].legend_.remove()

# Plot 3: Bound Tightness Comparison
sns.scatterplot(
    data=normData, x='Gershgorin Max', y='Brauer Max Product Root', 
    ax=axesM[0,2], color='coral', alpha=0.7, s=80
)
axesM[0,2].set(xscale='log', yscale='log', xlabel='Gershgorin Max Bound', ylabel='sqrt(Brauer Max Product)')
axesM[0,2].plot([1e-5, 1e10], [1e-5, 1e10], 'r--', alpha=0.5, label='1:1 Equivalency')
axesM[0,2].set_title("Bound Efficiency: Gershgorin vs Brauer Product", fontweight='bold', fontsize=13)
axesM[0,2].legend(loc='upper left', fontsize=9)

# Plot 4: The Feasibility Boundary
feasibilityData = analysisDf.dropna(subset=['Gershgorin Max', 'Minimum Singular Value (Original)'])
sns.scatterplot(
    data=feasibilityData, x='Gershgorin Max', y='Minimum Singular Value (Original)',
    hue='Kind_Grouped', palette=kindHues, s=80, alpha=0.7, ax=axesM[1,0]
)
axesM[1,0].set(xscale='log', yscale='symlog', linthresh=1e-5, xlabel='Gershgorin Max Bound', ylabel='Minimum Singular Value (Original)')
axesM[1,0].set_title("The Feasibility Boundary: Max Bound vs Min SVD", fontweight='bold', fontsize=13)
axesM[1,0].set_ylim(bottom=-1e-5)
axesM[1,0].legend_.remove()

# Plot 5: The Precision Limit
sns.scatterplot(data=analysisDf, x='Gershgorin Span', y='Condition Number (Original)', hue='Rank Deficient (Original)', palette='Set1', ax=axesM[1,1], alpha=0.7, s=80)
axesM[1,1].set(xscale='log', yscale='log', xlabel='Gershgorin Span (Algebraic Variant)')
axesM[1,1].set_title("The Precision Limit: Gershgorin Span vs Cond. Number", fontweight='bold', fontsize=13)

# Plot 6: Convexity Failure
fiedlerCondData = analysisDf.dropna(subset=['Brauer Max Product Root', 'Condition Number (Original)'])
sns.scatterplot(data=fiedlerCondData, x='Brauer Max Product Root', y='Condition Number (Original)', hue='Rank Deficient (Original)', palette='Set1', ax=axesM[1,2], alpha=0.7, s=80)
axesM[1,2].set(xscale='log', yscale='log', xlabel='sqrt(Brauer Max Product)')
axesM[1,2].set_title("Convexity Breakdown: Brauer Ceiling vs Cond. Number", fontweight='bold', fontsize=13)

# Plot 7: Structural Bias Impact
biasData = analysisDf.dropna(subset=['Directional Mean Bias (Laplacian)', 'Condition Number (Original)'])
sns.scatterplot(data=biasData, x='Directional Mean Bias (Laplacian)', y='Condition Number (Original)', hue='Rank Deficient (Original)', palette='Set1', ax=axesM[2,0], alpha=0.7, s=80)
axesM[2,0].set(xscale='log', yscale='log', xlabel='Directional Mean Bias')
axesM[2,0].set_title("KKT Bias: Pos/Neg Imbalance vs Cond. Number", fontweight='bold', fontsize=13)

# Plot 8: KKT Symmetry Breakdown
frobData = analysisDf.dropna(subset=['Signed Frobenius Ratio (Laplacian)', 'Condition Number (Original)'])
sns.scatterplot(data=frobData, x='Signed Frobenius Ratio (Laplacian)', y='Condition Number (Original)', hue='Rank Deficient (Original)', palette='Set1', ax=axesM[2,1], alpha=0.7, s=80)
axesM[2,1].set(xscale='symlog', yscale='log', linthresh=1e-4, xlabel='Signed Frobenius Ratio')
axesM[2,1].set_title("Symmetry Loss: Frobenius Ratio vs Cond. Number", fontweight='bold', fontsize=13)

# Plot 9: Bivariate Instability KDE
densityData = analysisDf.dropna(subset=['Condition Number (Original)', 'Gershgorin Span', 'Rank Deficient (Original)']).copy()
densityData['Condition Number (Log)'] = np.log10(densityData['Condition Number (Original)'].clip(lower=1e-10))
densityData['Gershgorin Span (Log)'] = np.log10(densityData['Gershgorin Span'].clip(lower=1e-10))
sns.kdeplot(
    data=densityData, x='Gershgorin Span (Log)', y='Condition Number (Log)', common_norm=False,
    hue='Rank Deficient (Original)', fill=True, alpha=0.4, palette='Set1', thresh=0.05, ax=axesM[2,2]
)
sns.scatterplot(
    data=densityData, x='Gershgorin Span (Log)', y='Condition Number (Log)',
    hue='Rank Deficient (Original)', palette='Set1', alpha=0.8, legend=True, 
    ax=axesM[2,2], s=40, edgecolor='white', linewidth=0.5
)
axesM[2,2].set_xlabel('Log10 Gershgorin Span')
axesM[2,2].set_ylabel('Log10 Condition Number (Original)')
axesM[2,2].set_title("Bivariate Map: Span vs Instability", fontweight='bold', fontsize=13)
sns.move_legend(axesM[2,2], "upper left", title='Rank Status', fontsize=9)

plt.tight_layout()
plt.show()


figC, axesC = plt.subplots(2, 3, figsize=(24, 20))

# Plot 1: Crossover Spearman Heatmap
cross_L = ['RCM Compression Ratio', 'Fiedler Value (Laplacian)', 'Brauer Max Center Distance (Laplacian)', 'Isolated Node Fraction']
cross_M = ['Directional Mean Bias (Laplacian)', 'Signed Frobenius Ratio (Laplacian)', 'Gershgorin Span', 'Brauer Max Product Root']
crossoverData = analysisDf[cross_L + cross_M].replace([np.inf, -np.inf], np.nan).dropna()
crossover_corr = crossoverData.corr(method='spearman').loc[cross_L, cross_M]
crossover_corr.index = [idx.replace(' (Laplacian)', '').replace('Brauer Max Center Distance', 'Brauer Max\nCenter Dist').replace('Isolated Node Fraction', 'Isolated\nNode Fraction') for idx in crossover_corr.index]
crossover_corr.columns = [col.replace(' (Laplacian)', '').replace('Directional Mean Bias', 'Dir. Mean\nBias').replace('Signed Frobenius Ratio', 'Signed\nFrob. Ratio').replace('Brauer Max Product Root', 'Brauer Root') for col in crossover_corr.columns]
sns.heatmap(crossover_corr, annot=True, cmap='PRGn', center=0, fmt=".2f", 
            vmin=-1, vmax=1, ax=axesC[0,0], cbar_kws={'label': 'Spearman Correlation'})
axesC[0,0].set_title('1. Crossover Interplay: Topology vs Algebra Matrix', fontweight='bold', fontsize=13)
axesC[0,0].tick_params(axis='x', rotation=45)

# Plot 2: Hub-Span Multiplier
fitHub = hubData[(hubData['Brauer Max Center Distance (Laplacian)'] > 0) & (hubData['Gershgorin Span'] > 0)]
m, c = np.polyfit(np.log10(fitHub['Brauer Max Center Distance (Laplacian)']), np.log10(fitHub['Gershgorin Span']), 1)
xLine = np.logspace(np.log10(fitHub['Brauer Max Center Distance (Laplacian)'].min()), np.log10(fitHub['Brauer Max Center Distance (Laplacian)'].max()), 100)
axesC[0,1].plot(xLine, (10**c) * (xLine**m), color='red', linestyle='--', linewidth=2, alpha=0.8)
sns.scatterplot(data=hubData, x='Brauer Max Center Distance (Laplacian)', y='Gershgorin Span', 
    hue='Rank Deficient (Original)', palette='Set1', alpha=0.7, s=60, ax=axesC[0,1]
)
axesC[0,1].set(xscale='log', yscale='log', xlabel='Brauer Max Center Distance (Topological Shift)')
axesC[0,1].set_title('2. The Hub Multiplier: Graph Shift vs Algebraic Variance', fontweight='bold', fontsize=13)

# Plot 3: Dead Weight vs. Bound Efficiency
validBounds = analysisDf.dropna(subset=['Isolated Node Fraction', 'Bound Looseness Ratio'])
sns.histplot(
    data=validBounds, x='Isolated Node Fraction', y='Bound Looseness Ratio',
    bins=30, pmax=0.9, cmap='mako_r', cbar=True, ax=axesC[0,2], cbar_kws={'label': 'Matrix Count'}
)
axesC[0,2].set(yscale='log', ylabel='Bound Looseness Ratio (Gershgorin / Brauer Root)')
axesC[0,2].set_title('3. Efficiency Map: Dead Weight vs Bound Looseness', fontweight='bold', fontsize=13)

# Plot 4: The Bias-Fracture Link (Bivariate KDE Contour)
biasKdeData = analysisDf.replace([np.inf, -np.inf], np.nan).dropna(subset=['Directional Mean Bias (Laplacian)', 'Fiedler Value (Laplacian)', 'Rank Deficient (Original)']).copy()
biasKdeData['Directional Mean Bias (Log)'] = np.log10(biasKdeData['Directional Mean Bias (Laplacian)'].clip(lower=1e-5))
biasKdeData['Fiedler (Log)'] = np.log10(biasKdeData['Fiedler Value (Laplacian)'].clip(lower=1e-8))
sns.kdeplot(
    data=biasKdeData, x='Directional Mean Bias (Log)', y='Fiedler (Log)', 
    hue='Rank Deficient (Original)', fill=True, alpha=0.4, palette='Set1', thresh=0.05, ax=axesC[1,0]
)
sns.scatterplot(
    data=biasKdeData, x='Directional Mean Bias (Log)', y='Fiedler (Log)', 
    hue='Rank Deficient (Original)', palette='Set1', alpha=0.8, legend=False, ax=axesC[1,0], s=30
)
axesC[1,0].set(xlabel='Log10 Directional Mean Bias (Algebraic)', ylabel='Log10 Fiedler Value (Topological)')
axesC[1,0].set_title('4. The Bias-Fracture Link: KKT Bias vs Connectivity', fontweight='bold', fontsize=13)

# Plot 5: Fracture State vs KKT Symmetry
violinData = analysisDf.replace([np.inf, -np.inf], np.nan).dropna(subset=['Is Disconnected', 'Signed Frobenius Ratio (Laplacian)']).copy()
sns.violinplot(
    data=violinData, x='Is Disconnected', y='Signed Frobenius Ratio (Laplacian)',
    hue='Rank Deficient (Original)', split=True, inner="quart", palette='Set1', ax=axesC[1,1]
)
axesC[1,1].set(yscale='symlog', linthresh=1e-4) # Automatically handles massive positive and negative ranges!
axesC[1,1].set_xlabel('Structurally Disconnected Graph? (Dmperm > 1)')
axesC[1,1].set_title('5. Fracture Distribution: Shattering vs KKT Mass', fontweight='bold', fontsize=13)

# Plot 6: Compression vs KKT Symmetry
symData = analysisDf.dropna(subset=['RCM Compression Ratio', 'Signed Frobenius Ratio (Laplacian)', 'Matrix Norm (Original)'])
sns.scatterplot(
    data=symData, x='RCM Compression Ratio', y='Signed Frobenius Ratio (Laplacian)', 
    hue='Kind_Grouped', palette=kindHues, size='Matrix Norm (Original)', 
    sizes=(20, 300), size_norm=mcolors.LogNorm(), alpha=0.7, ax=axesC[1,2]
)
axesC[1,2].set(xscale='log', yscale='symlog', linthresh=1e-4, ylabel='Signed Frobenius Ratio (Algebraic)', xlabel='RCM Compression Ratio (Topological)')
axesC[1,2].set_title('6. Structural Order vs KKT Symmetry', fontweight='bold', fontsize=13)
axesC[1,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Matrix Specs', fontsize=9)

plt.tight_layout()
plt.show()