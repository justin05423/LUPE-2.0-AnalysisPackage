import numpy as np
import matplotlib.pyplot as plt


def show_classifier_results(behavior_classes, all_score,
                            base_score, base_annot,
                            learn_score, learn_annot):

    keys = ['Behavior', 'Performance %', 'Iteration #']
    perf_by_class = {k: [] for k in behavior_classes}
    # all scores
    scores = np.vstack((np.hstack(base_score), np.vstack(learn_score)))
    # take means
    mean_scores = [100 * round(np.mean(scores[j], axis=0), 2) for j in range(len(scores))]
    # take the means of non-active learning
    mean_scores2beat = np.mean(all_score, axis=0)
    # make a copy
    scores2beat_byclass = all_score.copy()
    # for each behavior
    for c, c_name in enumerate(behavior_classes):
        if c_name != behavior_classes[-1]:
            for it in range(scores.shape[0]):
                perf_by_class[c_name].append(100 * round(scores[it][c], 2))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hlines(100 * mean_scores2beat, 0, len(mean_scores), ls='--', color='k')
    ax.plot(mean_scores)

    # print(mean_scores2beat, mean_scores)
    return fig, ax






