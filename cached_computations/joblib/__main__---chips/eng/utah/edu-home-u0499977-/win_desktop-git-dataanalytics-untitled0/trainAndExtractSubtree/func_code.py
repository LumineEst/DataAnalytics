# first line: 2562
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
