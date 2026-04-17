# first line: 1852
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
