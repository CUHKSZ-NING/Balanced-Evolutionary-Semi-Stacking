import math
import time
import numpy as np
import random
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score as gmean
from imblearn.ensemble import BalancedBaggingClassifier as BBC


class BalancedEvolutionaryCoTraining(object):
    def __init__(self, clf_1, clf_2=None, n_splits=5, population_size=10, iteration=10, learning_rate='auto',
                 sampling_ratio=0.1, meta_learner=BBC(base_estimator=SVC(probability=True), replacement=True),
                 evolve=True, verbose=True):
        self.clf_1_ = deepcopy(clf_1)
        if clf_2 is not None:
            self.clf_2_ = deepcopy(clf_2)
        else:
            self.clf_2_ = self.clf_1_
        self.n_splits_ = n_splits
        self.population_size_ = population_size
        self.iteration_ = iteration
        self.learning_rate_ = learning_rate
        self.sampling_ratio_ = sampling_ratio
        self.evolve_ = evolve
        self.verbose_ = verbose

        self.population = {}
        self.ga_paras = {}
        self.X_train = None
        self.y_train = None
        self.imbalance_ratio = 0
        self.num_features = 0
        self.learning_rate_base = 0

        self.X_positive = None
        self.X_negative = None
        self.X_unlabeled = None
        self.X_labeled = None
        self.y_labeled = None
        self.y_unlabeled = None

        # base_estimators: 1) multi-view; 2) single-view
        # base_samples: 1) safe samples; 2) pseudo samples
        # meta_samples: 1) safe samples; 2) pseudo samples

        # s: safe
        # u: unsafe
        # v: views

        self.meta_learner_ = {'default': deepcopy(meta_learner),  # suv
                              'safe': deepcopy(meta_learner),  # safe-stacking via only safe samples
                              'unsafe': deepcopy(meta_learner),  # safe-stacking via pseudo samples
                              'views': deepcopy(meta_learner),  # multi-view stacking via only safe samples
                              'pseudo': deepcopy(meta_learner),  # pseudo-stacking via only safe samples
                              'su': deepcopy(meta_learner),
                              'sv': deepcopy(meta_learner),
                              'uv': deepcopy(meta_learner),
                              }
        self.X_meta = {'default': None,
                       'safe': None,
                       'unsafe': None,
                       'views': None,
                       'pseudo': None,
                       'su': None,
                       'sv': None,
                       'uv': None,
                       }
        self.y_meta = {'default': None,
                       'pseudo': None}

        self.samples_need_label = 0

    def fit(self, X_train, y_train):
        time_start = time.time()
        self.initialize(X_train, y_train)
        self.evolutionary_cotraining()
        self.train_all(option='views')
        self.train_all(option='safe')
        self.train_all(option='unsafe')
        self.stacking()
        time_end = time.time()

        if self.verbose_:
            print('Training time: %.1fs.' % (time_end - time_start))
            print('Pseudo labels: {+%d, -%d}.\n' %
                  (len(np.where(self.y_unlabeled == 1)[0]), len(np.where(self.y_unlabeled == 0)[0])))
            pass

    def co_training(self):  # need update
        for key in self.population:
            if type(self.learning_rate_) == 'str':
                self.learning_rate_ = -1
                print('Invalid learning_rate')
            label_per_iter = len(self.X_unlabeled) * self.learning_rate_ / self.n_splits_ / 2 + self.samples_need_label
            self.samples_need_label = label_per_iter - int(label_per_iter)
            label_per_iter = int(label_per_iter)

            sampling = random.sample(range(0, len(self.X_unlabeled)),
                                     int(len(self.X_unlabeled) * self.sampling_ratio_))
            label_pred_1, label_pred_2 = self.predict_proba_one_split(key=key, X_test=self.X_unlabeled[sampling],
                                                                      option='views')
            score_positive = label_pred_1[..., 1] * label_pred_2[..., 1]
            score_negative = label_pred_1[..., 0] * label_pred_2[..., 0]
            pseudo_positive = score_positive.argsort()[:: -1][0: label_per_iter]
            pseudo_negative = score_negative.argsort()[:: -1][0: label_per_iter]

            for i in range(0, label_per_iter):
                if self.y_unlabeled[sampling[pseudo_positive[i]]] == -1:
                    self.y_unlabeled[sampling[pseudo_positive[i]]] = 1
                    for k in self.population:
                        self.population[k]['data']['X_positive'] = np.concatenate((
                            self.population[k]['data']['X_positive'], [self.X_unlabeled[sampling[pseudo_positive[i]]]]))
                if self.y_unlabeled[sampling[pseudo_negative[i]]] == -1:
                    self.y_unlabeled[sampling[pseudo_negative[i]]] = 0
                    for k in self.population:
                        self.population[k]['data']['X_negative'] = np.concatenate((
                            self.population[k]['data']['X_negative'], [self.X_unlabeled[sampling[pseudo_negative[i]]]]))

    def update(self):
        for key in self.population:
            units = self.population[key]['chromosome']
            for i in range(0, len(units)):
                samples_temp = np.zeros(len(self.population[key]['data']['X_negative']), dtype=int)
                samples_index = np.where(units[i]['samples'] == 1)[0]
                samples_temp[samples_index] = 1
                num_positive = len(self.population[key]['data']['X_positive'])
                num_negative = len(samples_index)

                if num_negative < num_positive:
                    samples_index = np.where(samples_temp == 0)[0]
                    samples_index_new = random.sample(range(0, len(samples_index)), num_positive - num_negative)
                    samples_temp[samples_index[samples_index_new]] = 1
                if num_negative > num_positive:
                    samples_index = np.where(samples_temp == 1)[0]
                    samples_index_new = random.sample(range(0, len(samples_index)), num_negative - num_positive)
                    samples_temp[samples_index[samples_index_new]] = 0

                units[i]['samples'] = deepcopy(samples_temp)

                if sum(units[i]['views']) in [0, len(units[i]['views'])]:
                    units[i]['views'] = np.zeros(self.num_features, dtype=int)
                    views_1 = np.array(random.sample(range(0, self.num_features),
                                                     int((self.num_features + random.randint(0, 1)) / 2)))
                    units[i]['views'][views_1] = 1

    def evolutionary_cotraining(self):
        for i in range(0, self.iteration_):
            self.train_all(option='views')
            if -1 in self.y_unlabeled:
                self.co_training()
            self.learning_rate_ = self.learning_rate_base * math.exp(i / self.iteration_)
            if self.evolve_ and self.population_size_ >= 2:
                self.get_fitness()
                self.cross_over()
                self.mutate()
            self.update()
            # print(len(np.where(self.y_unlabeled == 1)[0]), len(np.where(self.y_unlabeled == 0)[0]))

    def initialize(self, X_train, y_train):
        self.X_train = deepcopy(X_train)
        self.y_train = deepcopy(y_train)
        index_positive = np.where(y_train == 1)[0]
        index_negative = np.where(y_train == 0)[0]
        index_unlabeled = np.where(y_train == -1)[0]
        self.X_positive = deepcopy(self.X_train[index_positive])
        self.X_negative = deepcopy(self.X_train[index_negative])
        self.X_unlabeled = deepcopy(self.X_train[index_unlabeled])
        self.y_unlabeled = np.zeros(len(self.X_unlabeled), dtype=int) - 1
        num_positive = len(index_positive)
        num_negative = len(index_negative)
        # num_unlabeled = len(index_unlabeled)
        self.X_labeled = deepcopy(np.concatenate((self.X_positive, self.X_negative), axis=0))
        self.y_labeled = np.concatenate((np.ones(num_positive, dtype=int), np.zeros(num_negative, dtype=int)))

        skf = StratifiedKFold(n_splits=self.n_splits_)

        self.imbalance_ratio = num_negative / num_positive
        self.num_features = len(self.X_train[0, ...])

        if self.learning_rate_ == 'auto':
            self.learning_rate_base = 0.1 * (num_positive / (len(self.X_unlabeled) * self.imbalance_ratio + 1)) ** 0.5
            self.learning_rate_ = self.learning_rate_base

        key = 0
        for train_i, test_i in skf.split(self.X_labeled, self.y_labeled):
            self.population[key] = {}

            positive_i = np.where(self.y_labeled[train_i] == 1)[0]
            negative_i = np.where(self.y_labeled[train_i] == 0)[0]
            num_positive = len(positive_i)
            num_negative = len(negative_i)
            self.population[key]['data'] = {'X_positive': deepcopy(self.X_labeled[train_i][positive_i]),
                                            'X_negative': deepcopy(self.X_labeled[train_i][negative_i]),
                                            'X_test': deepcopy(self.X_labeled[test_i]),
                                            'y_test': deepcopy(self.y_labeled[test_i]),
                                            'safe_positive': num_positive,
                                            'safe_negative': num_negative
                                            }
            self.population[key]['chromosome'] = []
            for i in range(0, self.population_size_):
                self.population[key]['chromosome'].append({'clf_1': deepcopy(self.clf_1_),
                                                           'clf_2': deepcopy(self.clf_2_),
                                                           'clf_1_safe': deepcopy(self.clf_1_),
                                                           'clf_2_safe': deepcopy(self.clf_2_),
                                                           'clf_1_pseudo': deepcopy(self.clf_1_),
                                                           'clf_2_pseudo': deepcopy(self.clf_2_),
                                                           'views': np.zeros(self.num_features, dtype=int),
                                                           'samples': np.zeros(num_negative, dtype=int),
                                                           'fitness': 0,
                                                           })
                views_1 = np.array(random.sample(range(0, self.num_features),
                                                 int((self.num_features + random.randint(0, 1)) / 2)))
                self.population[key]['chromosome'][i]['views'][views_1] = 1
                random_i = np.array(random.sample(range(0, num_negative), num_positive))
                self.population[key]['chromosome'][i]['samples'][random_i] = 1

            key += 1

    def train_all(self, option):
        if option == 'views':  # views for co-training
            for key in self.population:
                X_positive = deepcopy(self.population[key]['data']['X_positive'])
                X_negative = deepcopy(self.population[key]['data']['X_negative'])
                for i in range(0, len(self.population[key]['chromosome'])):
                    unit = self.population[key]['chromosome'][i]
                    X_train = np.concatenate((X_positive, X_negative[np.where(unit['samples'] == 1)[0]]), axis=0)
                    y_train = np.concatenate((np.ones(len(X_positive), dtype=int),
                                              np.zeros(len(X_train) - len(X_positive), dtype=int)))
                    unit['clf_1'].fit(X_train[..., np.where(unit['views'] == 0)[0]], y_train)
                    unit['clf_2'].fit(X_train[..., np.where(unit['views'] == 1)[0]], y_train)
        if option == 'safe':  # safe prediction
            for key in self.population:
                safe_positive = self.population[key]['data']['safe_positive']
                safe_negative = self.population[key]['data']['safe_negative']
                X_positive = deepcopy(self.population[key]['data']['X_positive'][0: safe_positive])
                X_negative = deepcopy(self.population[key]['data']['X_negative'][0: safe_negative])
                for i in range(0, len(self.population[key]['chromosome'])):
                    unit = self.population[key]['chromosome'][i]
                    samples = np.where(unit['samples'] == 1)[0]
                    samples = resample(samples, safe_negative, safe_positive)
                    X_train = np.concatenate((X_positive, X_negative[samples]), axis=0)
                    y_train = np.concatenate((np.ones(len(X_positive), dtype=int),
                                              np.zeros(len(X_train) - len(X_positive), dtype=int)))
                    unit['clf_1_safe'].fit(X_train, y_train)
                    unit['clf_2_safe'].fit(X_train, y_train)
        if option == 'unsafe':  # semi prediction
            for key in self.population:
                X_positive = deepcopy(self.population[key]['data']['X_positive'])
                X_negative = deepcopy(self.population[key]['data']['X_negative'])
                for i in range(0, len(self.population[key]['chromosome'])):
                    unit = self.population[key]['chromosome'][i]
                    X_train = np.concatenate((X_positive, X_negative[np.where(unit['samples'] == 1)[0]]), axis=0)
                    y_train = np.concatenate((np.ones(len(X_positive), dtype=int),
                                              np.zeros(len(X_train) - len(X_positive), dtype=int)))
                    unit['clf_1_pseudo'].fit(X_train, y_train)
                    unit['clf_2_pseudo'].fit(X_train, y_train)

    def predict_proba_one_split(self, option, key=None, X_test=None):
        label_pred_1 = None
        label_pred_2 = None
        weight = len(self.population[key]['chromosome'])

        if X_test is None:
            X_test = deepcopy(self.population[key]['data']['X_test'])

        if option == 'views':  # multi-view prediction
            for i in range(0, len(self.population[key]['chromosome'])):
                unit = self.population[key]['chromosome'][i]
                if label_pred_1 is None:
                    label_pred_1 = deepcopy(unit['clf_1'].predict_proba(
                        X_test[..., np.where(unit['views'] == 0)[0]])) / weight
                    label_pred_2 = deepcopy(unit['clf_2'].predict_proba(
                        X_test[..., np.where(unit['views'] == 1)[0]])) / weight
                else:
                    label_pred_1 += deepcopy(unit['clf_1'].predict_proba(
                        X_test[..., np.where(unit['views'] == 0)[0]])) / weight
                    label_pred_2 += deepcopy(unit['clf_2'].predict_proba(
                        X_test[..., np.where(unit['views'] == 1)[0]])) / weight
        if option == 'safe':  # safe prediction
            for i in range(0, len(self.population[key]['chromosome'])):
                unit = self.population[key]['chromosome'][i]
                if label_pred_1 is None:
                    label_pred_1 = deepcopy(unit['clf_1_safe'].predict_proba(X_test)) / weight
                    label_pred_2 = deepcopy(unit['clf_2_safe'].predict_proba(X_test)) / weight
                else:
                    label_pred_1 += deepcopy(unit['clf_1_safe'].predict_proba(X_test)) / weight
                    label_pred_2 += deepcopy(unit['clf_2_safe'].predict_proba(X_test)) / weight
        if option == 'unsafe':  # unsafe prediction
            for i in range(0, len(self.population[key]['chromosome'])):
                unit = self.population[key]['chromosome'][i]
                if label_pred_1 is None:
                    label_pred_1 = deepcopy(unit['clf_1_pseudo'].predict_proba(X_test)) / weight
                    label_pred_2 = deepcopy(unit['clf_2_pseudo'].predict_proba(X_test)) / weight
                else:
                    label_pred_1 += deepcopy(unit['clf_1_pseudo'].predict_proba(X_test)) / weight
                    label_pred_2 += deepcopy(unit['clf_2_pseudo'].predict_proba(X_test)) / weight

        return label_pred_1, label_pred_2

    def stacking(self):
        X_pseudo = deepcopy(combine((self.X_unlabeled[np.where(self.y_unlabeled == 1)[0]],
                                     self.X_unlabeled[np.where(self.y_unlabeled == 0)[0]])))
        y_pseudo = combine((np.ones(len(np.where(self.y_unlabeled == 1)[0]), dtype=int),
                            np.zeros(len(np.where(self.y_unlabeled == 0)[0]), dtype=int)))
        X_pseudo_meta = None
        weight = len(self.population.keys())
        for key in self.population:
            label_pred = {0: (self.predict_proba_one_split(key=key, option='safe'))[0][..., 0].reshape(-1, 1),
                          1: (self.predict_proba_one_split(key=key, option='safe'))[1][..., 0].reshape(-1, 1),
                          2: (self.predict_proba_one_split(key=key, option='unsafe'))[0][..., 0].reshape(-1, 1),
                          3: (self.predict_proba_one_split(key=key, option='unsafe'))[1][..., 0].reshape(-1, 1),
                          4: (self.predict_proba_one_split(key=key, option='views'))[0][..., 0].reshape(-1, 1),
                          5: (self.predict_proba_one_split(key=key, option='views'))[1][..., 0].reshape(-1, 1),
                          6: (self.predict_proba_one_split(
                              key=key, X_test=X_pseudo, option='safe'))[0][..., 0].reshape(-1, 1),
                          7: (self.predict_proba_one_split(
                              key=key, X_test=X_pseudo, option='safe'))[1][..., 0].reshape(-1, 1)
                          }
            self.X_meta['default'] = combine((self.X_meta['default'],
                                              deepcopy(combine(label_pred, keys=[0, 1, 2, 3, 4, 5], axis=1))))
            self.X_meta['safe'] = combine((self.X_meta['safe'],
                                           deepcopy(combine(label_pred, keys=[0, 1], axis=1))))
            self.X_meta['unsafe'] = combine((self.X_meta['unsafe'],
                                             deepcopy(combine(label_pred, keys=[2, 3], axis=1))))
            self.X_meta['views'] = combine((self.X_meta['views'],
                                            deepcopy(combine(label_pred, keys=[4, 5], axis=1))))
            self.X_meta['su'] = combine((self.X_meta['su'],
                                         deepcopy(combine(label_pred, keys=[0, 1, 2, 3], axis=1))))
            self.X_meta['sv'] = combine((self.X_meta['sv'],
                                         deepcopy(combine(label_pred, keys=[0, 1, 4, 5], axis=1))))
            self.X_meta['uv'] = combine((self.X_meta['uv'],
                                         deepcopy(combine(label_pred, keys=[2, 3, 4, 5], axis=1))))
            if X_pseudo is not None:
                if X_pseudo_meta is None:
                    X_pseudo_meta = deepcopy(combine(label_pred, keys=[6, 7], axis=1)) / weight
                else:
                    X_pseudo_meta += deepcopy(combine(label_pred, keys=[6, 7], axis=1)) / weight

            self.y_meta['default'] = combine((self.y_meta['default'], deepcopy(self.population[key]['data']['y_test'])))

        self.meta_learner_['default'].fit(self.X_meta['default'], self.y_meta['default'])
        self.meta_learner_['safe'].fit(self.X_meta['safe'], self.y_meta['default'])
        self.meta_learner_['unsafe'].fit(self.X_meta['unsafe'], self.y_meta['default'])
        self.meta_learner_['views'].fit(self.X_meta['views'], self.y_meta['default'])
        self.meta_learner_['su'].fit(self.X_meta['su'], self.y_meta['default'])
        self.meta_learner_['sv'].fit(self.X_meta['sv'], self.y_meta['default'])
        self.meta_learner_['uv'].fit(self.X_meta['uv'], self.y_meta['default'])
        self.X_meta['pseudo'] = combine((self.X_meta['safe'], X_pseudo_meta))
        self.y_meta['pseudo'] = combine((self.y_meta['default'], y_pseudo))
        self.meta_learner_['pseudo'].fit(self.X_meta['pseudo'], self.y_meta['pseudo'])

    def predict_proba(self, X_test, strategy='default'):
        X_meta = None
        weight = len(self.population.keys())
        keys = None
        if strategy == 'default':
            keys = [0, 1, 2, 3, 4, 5]
        elif strategy == 'safe':
            keys = [0, 1]
        elif strategy == 'unsafe':
            keys = [2, 3]
        elif strategy == 'views':
            keys = [4, 5]
        elif strategy == 'pseudo':
            keys = [0, 1]
        elif strategy == 'su':
            keys = [0, 1, 2, 3]
        elif strategy == 'sv':
            keys = [0, 1, 4, 5]
        elif strategy == 'uv':
            keys = [2, 3, 4, 5]
        for key in self.population:
            label_pred = {0: (self.predict_proba_one_split(
                key=key, X_test=X_test, option='safe'))[0][..., 0].reshape(-1, 1),
                          1: (self.predict_proba_one_split(
                              key=key, X_test=X_test, option='safe'))[1][..., 0].reshape(-1, 1),
                          2: (self.predict_proba_one_split(
                              key=key, X_test=X_test, option='unsafe'))[0][..., 0].reshape(-1, 1),
                          3: (self.predict_proba_one_split(
                              key=key, X_test=X_test, option='unsafe'))[1][..., 0].reshape(-1, 1),
                          4: (self.predict_proba_one_split(
                              key=key, X_test=X_test, option='views'))[0][..., 0].reshape(-1, 1),
                          5: (self.predict_proba_one_split(
                              key=key, X_test=X_test, option='views'))[1][..., 0].reshape(-1, 1)}
            if X_meta is None:
                X_meta = deepcopy(combine(label_pred, keys=keys, axis=1)) / weight
            else:
                X_meta += deepcopy(combine(label_pred, keys=keys, axis=1)) / weight
        return self.meta_learner_[strategy].predict_proba(X_meta)

    def predict(self, X_test, strategy='default'):
        label_pred_proba = self.predict_proba(X_test, strategy)
        label_pred = np.zeros(len(label_pred_proba), dtype=int)
        positive_pred = np.where(label_pred_proba[..., 1] > 0.5)
        label_pred[positive_pred] = 1

        return label_pred

    def get_fitness(self):
        for key in self.population:
            X_test = deepcopy(self.population[key]['data']['X_test'])
            y_test = deepcopy(self.population[key]['data']['y_test'])
            for i in range(0, len(self.population[key]['chromosome'])):
                unit = self.population[key]['chromosome'][i]
                y_pred_1 = unit['clf_1'].predict(X_test[..., np.where(unit['views'] == 0)[0]])
                y_pred_2 = unit['clf_2'].predict(X_test[..., np.where(unit['views'] == 1)[0]])
                unit['fitness'] = gmean(y_test, y_pred_1) * gmean(y_test, y_pred_2)

    def cross_over(self):
        ratio_elite = 0.2
        ratio_cross = 0.2
        ratio_diverse = 0.3
        # ratio_mutate = 0.02

        num_elite = int(self.population_size_ * ratio_elite)

        population_new = {}
        for key in self.population:
            population_new[key] = []
            units = self.population[key]['chromosome']
            units.sort(key=lambda x: x['fitness'], reverse=True)
            # print('%.3f' % units[0]['fitness'])
            for i in range(0, len(units)):
                population_new[key].append({'views': None,
                                            'samples': None})

        for key in population_new:
            for i in range(0, self.population_size_):
                units = self.population[key]['chromosome']
                if i < num_elite:
                    population_new[key][i]['views'] = deepcopy(units[i]['views'])
                    population_new[key][i]['samples'] = deepcopy(units[i]['samples'])
                else:
                    pair = random.sample(range(0, int(self.population_size_)), 2)
                    vector_views = deepcopy(units[pair[0]]['views'])
                    vector_samples = deepcopy(units[pair[0]]['samples'])
                    for j in range(0, len(vector_views)):
                        if random.random() < ratio_cross:
                            vector_views[j] = units[pair[1]]['views'][j]
                    for j in range(0, len(vector_samples)):
                        if random.random() < ratio_cross:
                            vector_samples[j] = units[pair[1]]['samples'][j]
                        if random.random() < ratio_diverse:
                            key_elite = random.randint(0, num_elite - 1)
                            if vector_samples[j] == units[key_elite]['samples'][j]:
                                vector_samples[j] = np.abs(units[key_elite]['samples'][j] - 1)  # only 0 and 1

                    population_new[key][i]['views'] = deepcopy(vector_views)
                    population_new[key][i]['samples'] = deepcopy(vector_samples)

        for key in population_new:
            for i in range(0, self.population_size_):
                self.population[key]['chromosome'][i]['views'] = deepcopy(population_new[key][i]['views'])
                self.population[key]['chromosome'][i]['samples'] = deepcopy(population_new[key][i]['samples'])

    def mutate(self):
        pass


# Tools
def combine(data, keys=None, axis=0):  # A safe concatenate function
    if not (type(data) is tuple or type(data) is dict):
        return data

    data_temp = []

    for i in range(0, len(data)):
        if data[i] is not None and len(data[i]) > 0:
            if keys is not None and i not in keys:
                continue
            data_temp.append(deepcopy(data[i]))

    if len(data_temp) > 0:
        return np.concatenate(tuple(data_temp), axis=axis)
    else:
        return None


def resample(samples, limit_zone, limit_num):  # Resampling balanced safe data
    if type(samples) == list:
        samples = np.array(samples)

    num_keep = len(np.where(samples < limit_zone)[0])
    new_samples = limit_num - num_keep
    new_zone = np.zeros(limit_zone, dtype=int)
    new_zone[samples[np.where(samples < limit_zone)[0]]] = 1
    if new_samples > 0:
        change_zone = np.where(new_zone == 0)[0]
        changes = random.sample(range(0, len(change_zone)), new_samples)
        new_zone[change_zone[changes]] = 1
    elif new_samples == 0:
        pass
    else:
        change_zone = np.where(new_zone == 1)[0]
        changes = random.sample(range(0, len(change_zone)), new_samples * (-1))
        new_zone[change_zone[changes]] = 0

    return np.where(new_zone == 1)[0]
