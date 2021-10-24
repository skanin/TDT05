import sklearn
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import colorama
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np

X_train_pre = pd.read_csv('train_preprocessed_no_nan.csv')
X_test_pre = pd.read_csv('test_preprocessed_no_nan.csv')

X_train = X_train_pre.drop(['target', 'id'], axis=1)
y = X_train_pre['target']


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def split(X : pd.DataFrame, y : pd.DataFrame, size=.2):
    return train_test_split(X, y, test_size=size, stratify=y)


X_test = X_test_pre.copy()
X_train.drop(columns=['f9_2', 'f2', 'f4', 'f13', 'f22'], inplace=True)
X_test.drop(columns=['f9_2', 'f2', 'f4', 'f13', 'f22'], inplace=True)
X_train_split, X_test_split, y_train_split, y_test_split = split(X_train, y, size=.01)

n = len(X_train)

class_0 = len(X_train_pre[X_train_pre['target'] == 0])
class_1 = len(X_train_pre[X_train_pre['target'] == 1])


catboost = CatBoostClassifier( depth=1, learning_rate=1, iterations=2000, l2_leaf_reg = 1e-20, leaf_estimation_iterations=10, loss_function= 'CrossEntropy')

# fit(X_train, y, clf, eval_pool)
N_HYPEROPT_PROBES = 60
HYPEROPT_ALGO = tpe.suggest
colorama.init()

def get_catboost_params(space):
    params = dict()
    params['learning_rate'] = space['learning_rate']
    params['depth'] = int(space['depth'])
    params['l2_leaf_reg'] = space['l2_leaf_reg']
    params['border_count'] = space['border_count']
    #params['rsm'] = space['rsm']
    return params



obj_call_count = 0
cur_best_loss = np.inf
log_writer = open( 'catboost-hyperopt-log.txt', 'w' )




D_train = Pool(X_train, y)
D_test = Pool(X_test_split, y_test_split)


def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    print('\nCatBoost objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    params = get_catboost_params(space)

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str) )

    model = CatBoostClassifier(iterations=2000,
                                        learning_rate=params['learning_rate'],
                                        depth=int(params['depth']),
                                        loss_function='Logloss',
                                        use_best_model=True,
                                        task_type="CPU",
                                        eval_metric='AUC',
                                        l2_leaf_reg=params['l2_leaf_reg'],
                                        early_stopping_rounds=3000,
                                        od_type="Iter",
                                        border_count=int(params['border_count']),
                                        verbose=False
                                        )
    
    model.fit(D_train, eval_set=D_test, verbose=False)
    nb_trees = model.tree_count_

    print('nb_trees={}'.format(nb_trees))

    y_pred = model.predict_proba(D_test.get_features())
    test_loss = sklearn.metrics.log_loss(D_test.get_label(), y_pred, labels=[0, 1])
    acc = sklearn.metrics.accuracy_score(D_test.get_label(), np.argmax(y_pred, axis=1))
    auc = sklearn.metrics.roc_auc_score(D_test.get_label(), y_pred[:,1])

    log_writer.write('loss={:<7.5f} acc={} auc={} Params:{} nb_trees={}\n'.format(test_loss, acc, auc, params_str, nb_trees ))
    log_writer.flush()

    if test_loss<cur_best_loss:
        cur_best_loss = test_loss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)


    return{'loss':test_loss, 'status': STATUS_OK }


space = {
        'depth': hp.quniform("depth", 1, 6, 1),
        'border_count': hp.uniform ('border_count', 32, 255),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 8),
       }

trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=True)

print('-'*50)
print('The best params:')
print( best )
print('\n\n')