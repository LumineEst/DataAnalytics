# first line: 1110
    @cacheDir.cache
    def get_optimal_composite_tree(xData, yData, targetName):
        baseTree = DecisionTreeClassifier(random_state=42)
        gridSearch = GridSearchCV(baseTree, paramGrid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
        gridSearch.fit(xData, yData)
        return gridSearch.best_estimator_
