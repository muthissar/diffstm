from sklearn.metrics import precision_score
import scipy
import numpy as np
class Metrics:
    def precision(labels, predictions):
        return precision_score(y_true=labels, y_pred=predictions, average='micro')
    def precision_over_time(labels, predictions,max_t=2000):
        precisions = [] #[[] for _ in range()]
        labels_t = [[] for _ in range(max_t)]
        predictions_t = [[] for _ in range(max_t)]
        for label, prediction in zip(labels, predictions):
            for i in range(max_t):
                labels_t.append(label[i])
                predictions_t.append(prediction[i])
        #for i in range(len(labels))#range(labels.shape[1]):
        for label_t, prediction_t in zip(labels_t, predictions_t):
            precisions.append(Metrics.precision(label_t, prediction_t))
        return precisions
    def tp_stats(labels, predictions):
        tp = np.array(labels) == np.array(predictions)
        return tp.mean(), tp.std(ddof=1), len(tp)
    def t_test_from_stats(mean1, std1, nobs1, mean2, std2, nobs2):
        return scipy.stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, equal_var=False)
    def t_test(labels, predictions_a, predictions_b):
        tp_a = np.array(labels) == np.array(predictions_a)
        tp_b = np.array(labels) == np.array(predictions_b)
        return scipy.stats.ttest_ind(tp_a, tp_b, axis=0, equal_var=False, nan_policy='propagate') #not supported in current scipy version: , alternative='two-sided')