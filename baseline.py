from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from data import EuropolisFeatureDataset
from evaluation import average_all, average_class
import argparse
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prepare_europolis import text2new_label

from sklearn.utils import compute_class_weight

human_aq = ["cogency_human", "effectiveness_human", "reasonableness_human", "overall_human"]

feature_mapping = {
    'Word Count': "number of words",
    'negative_adjectives_component': "negative adjectives",
    'social_order_component': 'social order',
    'action_component': 'action verbs',
    'positive_adjectives_component': 'positive adjectives',
    'joy_component': 'joy adjectives',
    'affect_friends_and_family_component': 'affect/affiliation nouns',
    'fear_and_digust_component': 'fear/disgust',
    'politeness_component': 'politeness words',
    'polarity_nouns_component': 'polarity nouns',
    'polarity_verbs_component': 'polarity verbs',
    'virtue_adverbs_component': 'hostility/rectitude gain adverbs',
    'positive_nouns_component': 'positive nouns',
    'respect_component': 'respect nouns',
    'trust_verbs_component': 'trust/joy/positive verbs',
    'failure_component': 'power loss/failure verbs',
    'well_being_component': 'well-being words',
    'economy_component': 'economy words',
    'certainty_component': 'sureness nouns, quantity',
    'positive_verbs_component': 'positive verbs',
    'objects_component': 'objects',
    'mattr50_aw': 'type token ratio',
    'mtld_original_aw': 'TTR window-based',
    'hdd42_aw': 'lexical diversity',
    'MRC_Familiarity_AW': 'familiarity score',
    'MRC_Imageability_AW': 'imageability score',
    'Brysbaert_Concreteness_Combined_AW': 'concreteness score',
    'COCA_spoken_Range_AW': 'COCA Range norms',
    'COCA_spoken_Frequency_AW': 'COCA frequency norms',
    'COCA_spoken_Bigram_Frequency': '#bigrams spoken corpus',
    'COCA_spoken_bi_MI2': 'squared mutual information of bigrams',
    'McD_CD': 'Semantic variability contexts',
    'Sem_D': 'Co-occurrence probability',
    'All_AWL_Normed': 'academic words',
    'LD_Mean_RT': 'lexical decision reaction time',
    'LD_Mean_Accuracy': 'Average lexical decision accuracy',
    'WN_Mean_Accuracy': 'Average naming accuracy',
    'lsa_average_top_three_cosine': 'Average LSA cosine score',
    'content_poly': 'poylsemous content words',
    'hyper_verb_noun_Sav_Pav': 'hypernyms'
}


def get_feature_datasets(data_dir, quality_dim, text_col, i):
    train = EuropolisFeatureDataset(path_to_dataset=data_dir + "/split%i/train.csv" % i,
                                    label=quality_dim, text_col=text_col)
    val = EuropolisFeatureDataset(path_to_dataset=data_dir + "/split%i/val.csv" % i,
                                  label=quality_dim, text_col=text_col)
    test = EuropolisFeatureDataset(path_to_dataset=data_dir + "/split%i/test.csv" % i,
                                   label=quality_dim, text_col=text_col)
    return train, val, test


def get_class_names(label):
    """This method returns the class labels for a given DQ dimension"""
    name2class = text2new_label[label]
    class2name = dict(zip(name2class.values(), name2class.keys()))
    if label != "resp_gr":
        class_names = [class2name[i] for i in range(len(class2name))]
    else:
        class_names = ["disrespectful", "implicit respect", "explicit respect"]
    return class_names


def filter_features(dataset, add_AQ, add_linguistic):
    """Given a dataset, returns the dataset only containing the relevant features (linguistic, AQ or both)"""
    features_cols = []
    if add_linguistic:
        features_cols = [
            'Word Count', 'negative_adjectives_component', 'social_order_component', 'action_component',
            'positive_adjectives_component',
            'joy_component', 'affect_friends_and_family_component', 'fear_and_digust_component', 'politeness_component',
            'polarity_nouns_component', 'polarity_verbs_component', 'virtue_adverbs_component',
            'positive_nouns_component',
            'respect_component', 'trust_verbs_component', 'failure_component', 'well_being_component',
            'economy_component',
            'certainty_component', 'positive_verbs_component', 'objects_component', 'mattr50_aw', 'mtld_original_aw',
            'hdd42_aw', 'MRC_Familiarity_AW', 'MRC_Imageability_AW', 'Brysbaert_Concreteness_Combined_AW',
            'COCA_spoken_Range_AW', 'COCA_spoken_Frequency_AW', 'COCA_spoken_Bigram_Frequency', 'COCA_spoken_bi_MI2',
            'McD_CD', 'Sem_D', 'All_AWL_Normed', 'LD_Mean_RT', 'LD_Mean_Accuracy', 'WN_Mean_Accuracy',
            'lsa_average_top_three_cosine', 'content_poly', 'hyper_verb_noun_Sav_Pav']
    if add_AQ:
        for el in human_aq:
            features_cols.append(el)
    dataset = dataset[features_cols]
    # give the features more human readable names
    dataset.rename(feature_mapping, inplace=True, axis=1)
    return dataset


def plot_shap_values(classifier, dataset, class_names, output):
    """This method creates a tree explainer and computes the shap values for a given dataset"""

    # create the explainer
    explainer = shap.TreeExplainer(classifier)
    # compute the SHAP values for the given dataset
    shap_values = explainer.shap_values(dataset)

    # plot the overall summary plot
    f = plt.figure()
    shap.summary_plot(shap_values, dataset.values, plot_type="bar", class_names=class_names,
                      feature_names=dataset.columns, max_display=10)
    f.savefig("./plots/shap_plots/%s.png" % output, bbox_inches='tight',
              dpi=600)

    for i in range(len(class_names)):
        # plot a beeswarm plot for each class
        f = plt.figure()
        shap.summary_plot(shap_values[i], dataset, max_display=10)
        f.savefig("./plots/shap_plots/%s_%s.png" % (output, class_names[i]),
                  bbox_inches='tight', dpi=600)


def majority_baseline(train, test, label):
    """This method returns the result for a dummy baseline which always predicts the majorit label"""
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train, list(train[label]))
    test_y = list(test[label])
    predictions = dummy_clf.predict(test)
    report = classification_report(y_true=test_y, y_pred=predictions, output_dict=True)
    return report


def xg_boost(train, test, label, filter_feats, add_AQ, add_feats, plot_shap, i):
    """This method trains a boosted tree-ensemble"""
    # columns of the dataset
    feature_cols = list(train.num_cols)
    # create labels from train and test
    train_y = train.dataset[label]
    test_y = test.dataset[label]
    # extract the features that should be used for training
    if filter_feats:
        train_x = filter_features(train.dataset, add_AQ=add_AQ, add_linguistic=add_feats)
        test_x = filter_features(test.dataset, add_AQ=add_AQ, add_linguistic=add_feats)
    else:
        train_x = train.dataset[feature_cols]
        test_x = test.dataset[feature_cols]
    # set number of labels to three or 4 depending on the DQ dimension
    num_labels = 3
    if label == "jlev" or label == "int1":
        num_labels = 4
    # compute class weights to give more weight to less frequent classes
    weights = compute_class_weight(y=train_y, class_weight="balanced", classes=np.unique(train_y))
    weights_instance = [weights[i] for i in train_y]
    # create XGB classifier with default params and 100 estimators
    classifier = XGBClassifier(objective="multi:softmax", num_class=num_labels, n_estimators=100)
    classifier.fit(train_x, train_y, sample_weight=weights_instance)
    # create predictions for the test set
    y_pred = classifier.predict(test_x)
    # create shap plots if flag is true
    if plot_shap:
        plot_shap_values(classifier=classifier, class_names=get_class_names(label), dataset=test_x,
                         output="%s_all_classes_split%d" % (label, i))
    # return the classification report
    return classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dimension', type=str,
                        help='which quality dimension should be predicted')
    parser.add_argument("input_path", type=str, help="path to the director that contains the data for each split")
    parser.add_argument("addAQ", type=bool, action="store_true", help="whether to use human AQ scores as features")
    parser.add_argument("addFeats", type=bool, action="store_true", help="whether to use linguistic features.")
    parser.add_argument("plotShap", type=bool, action="store_true", help="whether to create shap plots.")
    args = parser.parse_args()

    test_reports_xgboost = []
    val_reports_xgboost = []
    test_reports_majority = []
    val_reports_majority = []
    for i in range(0, 5):
        train, dev, test = get_feature_datasets(
            data_dir=args.input_path,
            quality_dim=args.dimension,
            text_col="cleaned_comment", i=i)

        test_report = xg_boost(train, test, args.dimension, filter_feats=True, add_AQ=False, add_feats=True, i=i,
                               plot_shap=args.plotShap)
        val_report = xg_boost(train, test, args.dimension, filter_feats=True, add_AQ=args.addAQ,
                              add_feats=args.addFeats, i=i, plot_shap=args.plotShap)

        test_report_majority = majority_baseline(train, test, args.dimension)
        val_report_majority = majority_baseline(train, dev, args.dimension)

        test_reports_xgboost.append(test_report)
        val_reports_xgboost.append(val_report)

        test_reports_majority.append(test_report_majority)
        val_reports_majority.append(val_report_majority)

    label_list = list(set(train.labels))
    label_list = [str(l) for l in label_list]
    print("#######       majority baseline")
    print("#####   on validation:")
    print(average_all(val_reports_majority))
    print(average_class(val_reports_majority, label_list=label_list))
    print("#####   on test:")
    print(average_all(test_reports_majority))
    print(average_class(test_reports_majority, label_list=label_list))

    print("#######       tree-based")
    print("#####   on validation:")
    print(average_all(val_reports_xgboost))
    print(average_class(val_reports_xgboost, label_list=label_list))
    print("#####   on test:")
    print(average_all(test_reports_xgboost))
    print(average_class(test_reports_xgboost, label_list=label_list))
