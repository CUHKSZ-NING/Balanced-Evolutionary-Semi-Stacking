import numpy as np
import random
import joblib
import glob
from warnings import simplefilter
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.svm import SVC
from lightgbm import LGBMClassifier as LGBM
from BalancedEvolutionarySemiStacking import BalancedEvolutionarySemiStacking


def test_run():
    dataset_name = 'biomedical'
    # dataset_name = 'imb_datasets'  # extensive datasets

    data = joblib.load('dataset/' + dataset_name + '.pkl')

    for key in data:
        minor = len(np.where(data[key]['target'] == 1)[0])
        major = len(np.where(data[key]['target'] == 0)[0])
        attributes = len(data[key]['data'][0])
        print('%d / %d: %.2f (%d) <---- %s' % (minor, minor + major, major / minor, attributes, key))
    print('')

    n_rounds = 5
    k_fold = 5
    paras = {'ratio_train': 0.2,
             'balanced': False,
             'semi_supervised': True,
             'unlabeled_samples': False  # whether adding extra unlabeled data
             }

    results = {}

    for key in data:
        results[key] = {}
        for algorithm in ['BESS']:
            results[key][algorithm] = {}
            for r in range(0, n_rounds):
                results[key][algorithm][r] = []

    for r in range(0, n_rounds):
        skf = StratifiedKFold(n_splits=k_fold)
        for key in data:
            index_positive = np.where(data[key]['target'] == 1)[0]
            index_negative = np.where(data[key]['target'] == 0)[0]
            index_unlabeled = np.where(data[key]['target'] == -1)[0]
            X_labeled = deepcopy(data[key]['data'][np.concatenate((index_positive, index_negative))])
            y_labeled = np.concatenate((np.ones(len(index_positive), dtype=int),
                                        np.zeros(len(index_negative), dtype=int)))
            X_unlabeled = deepcopy(data[key]['data'][index_unlabeled])

            for index_train, index_test in skf.split(X_labeled, y_labeled):
                X_train = deepcopy(X_labeled[index_train])
                y_train = deepcopy(y_labeled[index_train])
                X_test = deepcopy(X_labeled[index_test])
                y_test = deepcopy(y_labeled[index_test])

                X_new, y_new = get_data(X_train, y_train, X_unlabeled=X_unlabeled, paras=paras)

                name = 'BECT'
                model = get_model(name=name, X_train=X_new, y_train=y_new)
                y_pred = model.predict(X_test)
                gmean = geometric_mean_score(y_test, y_pred)
                results[key]['BESS'][r].append(gmean)

            print_results(results, n_rounds)

    print('Final results after %d rounds of %d-fold CV:\n' % (n_rounds, k_fold))
    print_results(results, n_rounds)


def get_data(X_train, y_train, X_unlabeled=None, paras=None):
    ratio_train = paras['ratio_train']
    balanced = paras['balanced']
    semi_supervised = paras['semi_supervised']
    unlabeled_samples = paras['unlabeled_samples']

    index_positive = np.where(y_train == 1)[0]
    index_negative = np.where(y_train == 0)[0]

    num_positive = int(len(index_positive) * ratio_train)
    if balanced is True:
        num_negative = num_positive
    else:
        num_negative = int(len(index_negative) * ratio_train)

    random_positive = np.array(random.sample(range(0, len(index_positive)), num_positive))
    new_positive = index_positive[random_positive]
    unlabeled_positive = list(set(index_positive).difference(set(new_positive)))
    random_negative = np.array(random.sample(range(0, len(index_negative)), num_negative))
    new_negative = index_negative[random_negative]
    unlabeled_negative = list(set(index_negative).difference(set(new_negative)))

    X_new = deepcopy(X_train[np.concatenate((new_positive, new_negative))])
    y_new = np.concatenate((np.ones(len(new_positive), dtype=int), np.zeros(len(new_negative), dtype=int)))

    if semi_supervised is True:
        if unlabeled_samples is True:
            X_new = np.concatenate((X_new, X_unlabeled), axis=0)
            y_new = np.concatenate((y_new, np.zeros(len(X_unlabeled), dtype=int) - 1))

        if ratio_train < 1.0:
            X_semi = deepcopy(X_train[np.concatenate((unlabeled_positive, unlabeled_negative))])
            y_semi = np.zeros(len(X_semi), dtype=int) - 1
            X_new = np.concatenate((X_new, X_semi), axis=0)
            y_new = np.concatenate((y_new, y_semi))

    return X_new, y_new


def get_model(name, X_train, y_train):
    debug_info(name, y_train, sampling=None, semi_learner=None, paras_ablation=None)
    X_new = deepcopy(X_train)
    y_new = deepcopy(y_train)

    paras = {'clf_1': SVC(probability=True),
             'clf_2': LGBM(),
             'iteration': 10,
             'population_size': 10,
             'evolve': True}

    classifier = BalancedEvolutionarySemiStacking(
        clf_1=paras['clf_1'], clf_2=paras['clf_2'], iteration=paras['iteration'], evolve=paras['evolve'],
        population_size=paras['population_size'])
    classifier.fit(X_new, y_new)

    return classifier


def debug_info(name, y_train, sampling=None, semi_learner=None, debug=True, paras_ablation=None):
    if not debug:
        return

    num_positive = len(np.where(y_train == 1)[0])
    num_negative = len(np.where(y_train == 0)[0])
    num_unlabeled = len(np.where(y_train == -1)[0])

    print('Training via {%s, %s, %s} \n with {%d, %d, %d} samples. \n '
          % (name, sampling, semi_learner, num_positive, num_negative, num_unlabeled))

    if paras_ablation is not None:
        for key in paras_ablation:
            print('* \'%s\':' % key, paras_ablation[key])


def analysis_tongue(path, n_rounds):
    files = glob.glob(path + '*.pkl')
    data = {}
    for i in range(0, len(files)):
        print(files[i])
        data_raw = joblib.load(files[i])
        data[i] = print_results(data_raw, n_rounds=n_rounds, verbose=True)

    for algorithm in data[0]['DM']:
        for dataset in ['DM', 'CKD', 'BC']:
            for key in data:
                mean = data[key][dataset][algorithm]['mean']
                std = data[key][dataset][algorithm]['std']
                print('$%.1f _{\\pm %.2f}$ & ' % (100 * mean, 100 * std), end='')
                # print('%.1f, ' % (100 * mean), end='')
        print(algorithm)


def print_results(results, n_rounds, verbose=False):
    data = {}
    for key in results:
        if not verbose:
            print('')
            print(key)
        data[key] = {}
        for algorithm in results[key]:
            performance = []
            for r in range(0, n_rounds):
                metric_value = np.mean(results[key][algorithm][r])
                if metric_value == metric_value:
                    performance.append(np.mean(results[key][algorithm][r]))
            mean = np.mean(performance)
            std = np.std(performance, ddof=1)
            if mean == mean and not verbose:  # bool(nan == nan) is False
                print('$%.1f _{\\pm %.2f}$ <---- %s' % (100 * mean, 100 * std, algorithm))
            data[key][algorithm] = {'mean': mean, 'std': std}

    if not verbose:
        print('')

    return data


if __name__ == '__main__':
    simplefilter(action="ignore", category=FutureWarning)
    simplefilter(action="ignore", category=RuntimeWarning)
    test_run()
