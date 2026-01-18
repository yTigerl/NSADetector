from collections import defaultdict
from classificationAggregator import buildeComparePairs

def classify_bundle(T, reference_set, model, S, K=10, theta=2, major=5): #epsilon=0.2
    scores = []
    sample1_x, sample1_a, sample2_x, sample2_a = buildeComparePairs(T, reference_set)
    predict = model.predict([sample1_x, sample1_a, sample2_x, sample2_a])
    # for idx, T_ref in enumerate(reference_set):
    #     cls_id = idx // S
    #     score = model([T, T_ref])  # dissimilarity score
    #     scores.append((score, cls_id))
    for idx, score in enumerate(predict):
        cls_id = idx // S
        scores.append((score, cls_id))

    # Sort by similarity score
    scores.sort()
    top_k = scores[:K]

    # Voting
    vote_count = defaultdict(int)
    for score, cls_id in top_k:
        if score < theta:
            vote_count[cls_id] += 1

    top_vote = max(vote_count.values(), default=0)
    score_range = top_k[-1][0] - top_k[0][0]

    if top_vote < (K // 2 + 1):   # or score_range > epsilon
        return -1
    else:
        return max(vote_count, key=vote_count.get)  # 返回最常见类别
