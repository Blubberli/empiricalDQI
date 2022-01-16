import statistics


def average_all(reports):
    """Get a string of the average of the global scores (Fmacro, Accuracy), given a list of classification reports."""
    s = "metric\tmean\tdeviation\n"
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    for m in metrics:
        if m == 'accuracy':
            all_vals = [report[m] for report in reports]
        else:
            all_vals = [report['macro avg'][m] for report in reports]
        s += "%s\t%.2f\t%.2f\n" % (m, statistics.mean(all_vals), statistics.stdev(all_vals))
    return s


def average_class(reports, label_list):
    """Get a string of the average of the performance for each class, given a list of classification reports."""
    s = "label\tmetric\tmean\tdeviation\n"
    label_list = [str(l) for l in label_list]
    for metric in reports[0]['0'].keys():
        for label in label_list:
            all_vals = [report[label][metric] for report in reports]
            s += "%s\t%s\t%.2f\t%.2f\n" % (label, metric, statistics.mean(all_vals), statistics.stdev(all_vals))
    return s
