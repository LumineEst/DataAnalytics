# first line: 884
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
