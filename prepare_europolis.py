import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
import numpy as np
from preprocessing import clean_comment

random.seed(41)
# merge the classes and give them meaningful names(?)
# plot the class distribution(?)
# have german, english, original language text, all preprocessed, have unique ID, have argQuality Scores
# create a stratified split for each quality dimension

label2textencoding = {
    "jlev": {
        0: 'no justification',
        1: 'inferior justification',
        2: 'qualified justification',
        3: 'sophisticated (broad)',
        4: 'sophisticated (depth)'

    },
    "jcon": {
        0: 'own country',
        1: 'no reference',
        2: 'reference to common good (EU)',
        3: 'reference to common good (solidarty)'

    },
    "resp_gr": {
        0: 'disrespectful',
        1: 'implicit respect',
        2: 'balanced respect',
        3: 'explicit respect'
    },
    "int1": {
        0: 'no reference',
        1: 'negative reference',
        2: 'neutral reference',
        3: 'positive reference'

    }
}

text2new_label = {
    "jlev": {
        'no justification': 0,
        'inferior justification': 1,
        'qualified justification': 2,
        'sophisticated (broad)': 3,
        'sophisticated (depth)': 3

    },
    "jcon": {
        'own country': 0,
        'no reference': 1,
        'common good': 2,
        'common good': 2

    },
    "resp_gr": {
        'disrespectful': 0,
        'implicit respect': 1,
        'balanced respect': None,
        'explicit respect': 2
    },
    "int1": {
        'no reference': 0,
        'negative reference': 1,
        'neutral reference': 2,
        'positive reference': 3

    }

}


def convert_label_column(dimension, label_list):
    """Convert the old labels to the new labels and return the new label list"""
    text_labels = [label2textencoding[dimension][l] for l in label_list]
    new_labels = [text2new_label[dimension][text_label] for text_label in text_labels]
    return new_labels


def plot_class_distributions():
    """Read the whole dataset and plot label distribution for each quality dimension"""
    data = pd.read_csv("data/europolis_newDQI.csv", sep="\t")
    dimensions = ["jlev", "jcon", "resp_gr", "int1"]
    for dim in dimensions:
        counts = pd.crosstab(index=data[dim], columns='count')
        counts.plot.bar()
        plt.savefig("data/plots/original_class_distribution/%s.png" % dim)


def plot_class_distribution_split(split_number, input_dir, output_dir):
    dimensions = ["jlev", "jcon", "resp_gr", "int1"]
    for dim in dimensions:
        training_data = pd.read_csv("%s/%s/split%d/train.csv" % (input_dir, dim, split_number), sep="\t")
        dev_data = pd.read_csv("%s/%s/split%d/val.csv" % (input_dir, dim, split_number), sep="\t")
        test_data = pd.read_csv("%s/%s/split%d/test.csv" % (input_dir, dim, split_number), sep="\t")
        counts = pd.crosstab(index=training_data[dim], columns='count')
        counts.plot.bar()
        plt.savefig(
            "%s/%s_train.png" % (output_dir, dim))
        counts = pd.crosstab(index=dev_data[dim], columns='count')
        counts.plot.bar()
        plt.savefig(
            "%s/%s_val.png" % (output_dir, dim))
        counts = pd.crosstab(index=test_data[dim], columns='count')
        counts.plot.bar()
        plt.savefig(
            "%s/%s_test.png" % (output_dir, dim))


def create_stratified_split(quality_dim, output_dir):
    """Given a label and the output path, generate a 5fold (stratified) split"""
    data = pd.read_csv("data/europolis_newDQI.csv", sep="\t")
    # drop where label is None
    data = data[data[quality_dim].notna()]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    counter = 0
    for train, test in kfold.split(data, data[quality_dim]):
        train_set, test_set = data.loc[train], data.loc[test]
        train_set, val_set = train_test_split(train_set, test_size=0.25)

        train_set.to_csv("%s/split%d/train.csv" % (output_dir, counter),
                         sep="\t", index=False)
        val_set.to_csv("%s/split%d/val.csv" % (output_dir, counter), sep="\t",
                       index=False)
        test_set.to_csv("%s/split%d/test.csv" % (output_dir, counter),
                        sep="\t", index=False)
        counter += 1

        print("train: %d, val: %d, test:%d" % (len(train_set), len(val_set), len(test_set)))
        print('train: %.2f, val:%.2f, test: %.2f' % (
            len(train_set) / len(data), len(val_set) / len(data), len(test_set) / len(data)))


def check_no_overlap_train_test(quality_dim):
    input_dir = "data/5foldStratified/%s" % quality_dim
    for i in range(0, 5):
        training_data = pd.read_csv("%s/split%d/train.csv" % (input_dir, i), sep="\t")
        test_data = pd.read_csv("%s/split%d/test.csv" % (input_dir, i), sep="\t")
        val_data = pd.read_csv("%s/split%d/val.csv" % (input_dir, i), sep="\t")
        print(set(training_data.ID).intersection(test_data.ID))
        print(set(training_data.ID).intersection(val_data.ID))
        print(set(training_data.cleaned_comment).intersection(test_data.cleaned_comment))
        print(set(training_data.cleaned_comment).intersection(val_data.cleaned_comment))


def create_augmented_training_data(input_dir, quality_dim, output_dir):
    augmented_data = pd.read_csv("/Users/falkne/PycharmProjects/DQI/data/europolis_augmented.csv", sep="\t",
                                 header=None, names=["ID", "cleaned_comment"])
    input_dir = "%s/%s" % (input_dir, quality_dim)
    for i in range(0, 5):
        training_data = pd.read_csv("%s/split%d/train.csv" % (input_dir, i), sep="\t")
        # copy the training data and remove the original text column
        training_data_copy = training_data.copy()
        training_data_copy.drop(columns=['cleaned_comment'], inplace=True)
        # read test and dev
        test_data = pd.read_csv("%s/split%d/test.csv" % (input_dir, i), sep="\t")
        val_data = pd.read_csv("%s/split%d/val.csv" % (input_dir, i), sep="\t")

        # extract all possible target comments (based on what ID we have in the training data)
        available_data = augmented_data[augmented_data.ID.isin(training_data.ID)]
        # extract all other column values based on the original comment (from training data, based on ID)
        target_augmented = pd.merge(available_data, training_data_copy, on="ID")
        # add a column that indicates whether the comment was augmented or not
        target_augmented["augmented"] = [True] * len(target_augmented)
        training_data["augmented"] = [False] * len(training_data)
        training_dist = pd.crosstab(index=training_data[quality_dim], columns='count').to_dict()["count"]
        augmented_dist = pd.crosstab(index=target_augmented[quality_dim], columns='count').to_dict()["count"]
        # 1) if possible sample from lower frequency classes such that they match the most frequent class
        highest_existing_val = max(training_dist.values())
        print("largest training freq is %d" % highest_existing_val)
        print("smallest augmented freq is %d" % min(augmented_dist.values()))
        sampled_augmented_data = []
        # if frequencies in the augmented data are lower than the training freq of most frequent class, upsample lower_freq classes as much as possible
        if min(augmented_dist.values()) < highest_existing_val:
            for label, frequency in augmented_dist.items():
                if frequency < highest_existing_val:
                    sample = target_augmented[target_augmented[quality_dim] == label]
                else:
                    nr_additional_comments = highest_existing_val - training_dist[label]
                    sample = target_augmented[target_augmented[quality_dim] == label].sample(nr_additional_comments,
                                                                                             ignore_index=True)
                sampled_augmented_data.append(sample)
            augmented_all = pd.concat(sampled_augmented_data)
        # balance class frequencies first
        else:
            for label, frequency in augmented_dist.items():
                nr_additional_comments = highest_existing_val - training_dist[label]
                sample = target_augmented[target_augmented[quality_dim] == label].sample(nr_additional_comments,
                                                                                         ignore_index=True)
                sampled_augmented_data.append(sample)
            # now all classes should have the same frequency
            sampled_augmented_data_all = pd.concat(sampled_augmented_data)
            # remove the data that is already in the sample
            target_augmented = target_augmented[
                ~target_augmented['cleaned_comment'].isin(sampled_augmented_data_all['cleaned_comment'])]
            # add as much more data as possible to keep classes balanced
            # class frequency in the left augmented
            leftover_augmented_freqs = pd.crosstab(index=target_augmented[quality_dim], columns='count').to_dict()[
                "count"]
            # the current augmented data should be balanced now
            current_class_freq = pd.crosstab(index=pd.concat([sampled_augmented_data_all, training_data])[quality_dim],
                                             columns='count').to_dict()["count"]
            # take the minimum leftover samples and add them
            smallest_possible_val = min(leftover_augmented_freqs.values())
            new_samples = []
            for label, frequency in leftover_augmented_freqs.items():
                sample = target_augmented[target_augmented[quality_dim] == label].sample(smallest_possible_val,
                                                                                         ignore_index=True)
                new_samples.append(sample)
            sampled_augmented_data_all2 = pd.concat(new_samples)
            augmented_all = pd.concat([sampled_augmented_data_all, sampled_augmented_data_all2])
        augmented_training_data = pd.concat([training_data, augmented_all])
        augmented_training_data.to_csv("%s/%s/split%d/train.csv" % (output_dir, quality_dim, i), sep="\t", index=False)
        print(pd.crosstab(index=augmented_training_data[quality_dim], columns='count').to_dict()["count"])
        val_data.to_csv("%s/%s/split%d/val.csv" % (output_dir, quality_dim, i), sep="\t", index=False)
        test_data.to_csv("%s/%s/split%d/test.csv" % (output_dir, quality_dim, i), sep="\t", index=False)


def create_features_splits():
    dims = ["jlev", "jcon", "int1", "resp_gr"]
    europolis_with_feats = pd.read_csv("/Users/falkne/PycharmProjects/DQI/data/europolis_with_features.csv", sep="\t")
    dropped = False
    oldpath = "/Users/falkne/PycharmProjects/DQI/data/5foldStratified"
    newpath = "/Users/falkne/PycharmProjects/DQI/data/5foldFeatures"
    for dim in dims:
        for i in range(0, 5):

            train = pd.read_csv("%s/%s/split%d/train.csv" % (oldpath, dim, i), sep="\t")
            val = pd.read_csv("%s/%s/split%d/val.csv" % (oldpath, dim, i), sep="\t")
            test = pd.read_csv("%s/%s/split%d/test.csv" % (oldpath, dim, i), sep="\t")
            if not dropped:
                to_be_dropped = train.columns
                to_be_dropped = [el for el in to_be_dropped if el != "ID" and el in europolis_with_feats.columns]
                europolis_with_feats = europolis_with_feats.drop(columns=to_be_dropped)
                dropped = True
            train = pd.merge(train, europolis_with_feats, on="ID")
            val = pd.merge(val, europolis_with_feats, on="ID")
            test = pd.merge(test, europolis_with_feats, on="ID")
            train.to_csv("%s/%s/split%d/train.csv" % (newpath, dim, i), index=False, sep="\t")
            val.to_csv("%s/%s/split%d/val.csv" % (newpath, dim, i), index=False, sep="\t")
            test.to_csv("%s/%s/split%d/test.csv" % (newpath, dim, i), index=False, sep="\t")


def get_dataset_sizes(path, dim):
    lengths = []
    val_lengths = []
    test_length = []
    for i in range(0, 5):
        training = pd.read_csv("%s/%s/split%d/train.csv" % (path, dim, i), sep="\t")
        val = pd.read_csv("%s/%s/split%d/val.csv" % (path, dim, i), sep="\t")
        test = pd.read_csv("%s/%s/split%d/test.csv" % (path, dim, i), sep="\t")
        val_lengths.append(len(val))
        test_length.append(len(test))
        lengths.append(len(training))
    print("length for %s : %d" % (dim, np.average(np.array(lengths))))
    print("length for val : %d" % (np.average(np.array(val_lengths))))
    print("length for test : %d" % (np.average(np.array(test_length))))
