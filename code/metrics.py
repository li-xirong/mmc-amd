from sklearn import metrics


def sensitivity_score(expects, predicts):
    """just support binary-classes"""
    conf_mat = metrics.confusion_matrix(expects, predicts)
    return float(conf_mat[1, 1]) / float(conf_mat[1, 1] + conf_mat[1, 0])


def specificity_score(expects, predicts):
    """just support binary-classes"""
    conf_mat = metrics.confusion_matrix(expects, predicts)
    return 1 - float(conf_mat[0, 1]) / float(conf_mat[0, 1] + conf_mat[0, 0])


def f1_score(A, B):
    return 2.0 * A * B / (A + B)


def accuracy_score(expects, predicts):
    return metrics.accuracy_score(expects, predicts)


def confusion_matrix(expects, predicts):
    return metrics.confusion_matrix(expects, predicts)
