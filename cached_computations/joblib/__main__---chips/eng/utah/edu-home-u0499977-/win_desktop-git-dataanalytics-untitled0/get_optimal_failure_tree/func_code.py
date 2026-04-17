# first line: 1015
    @cacheDir.cache
    def get_optimal_failure_tree(xData, yData, targetName):
        baseTree = DecisionTreeClassifier(random_state=42)
        mcCv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
        gridSearch = GridSearchCV(baseTree, paramGrid, cv=mcCv, scoring='balanced_accuracy', n_jobs=-1)
        gridSearch.fit(xData, yData)
        return gridSearch.best_estimator_, gridSearch.best_score_
