def black_box_function(x,y):

    return -x**2 - (y-0) **2 +1
import numpy as np

x_range = np.linspace(-100, 100, num=1000) # -100 ~ 100 사이의 임의의 x를 만들어 냅니다.

y_range = np.linspace(-100, 100, num=1000) # -100 ~ 100 사이의 임의의 y를 만들어 냅니다.



# 미리 지정해둔 함수에 출력값을 받아 그림으로 확인하겠습니다 .
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(111, projection='3d')



X, Y = np.meshgrid(x_range, y_range)

Z = black_box_function(X,Y)

ax = plt.axes(projection='3d')

ax.contour3D(X, Y, Z, 50, cmap='binary')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')

ax.view_init(50, 30)
# 2. Getting Started

from bayes_opt import BayesianOptimization



# Bounded region of parameter space

pbounds = {'x': (-10, 10), 'y': (-10, 10)}



# 세부 사항 설정

optimizer = BayesianOptimization(

    f=black_box_function,

    pbounds=pbounds,

    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent

    random_state=1

)



# 최대화!!

optimizer.maximize(init_points=2, n_iter=30 )

    # n_iter: 반복 횟수 (많을 수록 정확한 값을 얻을 수 있다)

    # init_points: 초기 랜덤 포인트 갯수
# 최적의 (x, y) 값 결과 확인

print(optimizer.max)
optimizer.res[-5:] # 이전 history 를 확인 할 수 있다.
# 2.1 Changing bounds

optimizer.set_bounds(new_bounds={"x": (-1, 1)})



# 이후 절차는 동일히다.

optimizer.maximize( init_points=0, n_iter=5)
# 최적의 파라미터 값 확인

print(optimizer.max)
# 모듈 불러오기

from bayes_opt import BayesianOptimization

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score

import numpy as np



# seed 고정!

np.random.seed(0)

n_samples, n_features = 100, 1 # 100개의 데이터와 1개의 변수 생성



X = np.random.randn(n_samples, n_features) # shape = (100 , 1)

y = np.random.randn(n_samples) # shape = (100 , )





# CV 를 이용한, Ridge 파라미터 찾기

def Ridge_cv(alpha):

    '''

    :param alpha: Ridge's 하이퍼 파라미터

    :return: -RMSE --> 최소화를 위해 음수 부호를 붙힘

    '''



    RMSE = cross_val_score(Ridge(alpha=alpha), X, y, scoring='neg_mean_squared_error', cv=5).mean()



    return -RMSE
# 파라미터를 탐색할 공간

# Ridge는 0 ~ 10 사이에서 적절한 값을 찾는다.

pbounds = {'alpha': ( 0, 10 )}



# 베이지안 옵티마이제이션 객체를 생성

Ridge_BO = BayesianOptimization( f = Ridge_cv, pbounds  = pbounds , verbose=2, random_state=1 )



# 최대화!!!

Ridge_BO.maximize(init_points=2, n_iter = 10)



Ridge_BO.max # 찾은 파라미터 값 확인
import numpy as np

import matplotlib

from matplotlib import pyplot  as plt

from sklearn import svm, datasets

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization



matplotlib.rc('font', family = 'Malgun Gothic')



iris = datasets.load_iris()

X = iris.data

y = iris.target



def SVM_rbf_cv(gamma, C):

    model = svm.SVC(kernel = 'rbf', gamma=gamma, C = C)

    RMSE = cross_val_score(model, X, y, scoring='accuracy', cv=5).mean()

    return -RMSE
# 주어진 범위 사이에서 적절한 값을 찾는다.

pbounds = {'gamma': ( 0.001, 1000 ), "C": (0.001, 1000)}



# 베이지안 옵티마이제이션 객체를 생성

SVM_rbf_BO = BayesianOptimization( f = SVM_rbf_cv, pbounds = pbounds, verbose = 2, random_state = 1 )



# 메소드를 이용해 최대화!

SVM_rbf_BO.maximize(init_points=2, n_iter = 10)



SVM_rbf_BO.max # 찾은 파라미터 값 확인
import numpy as np

import matplotlib

from matplotlib import pyplot as plt

from sklearn import datasets

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization

import xgboost as xgb





iris = datasets.load_iris()

X = iris.data

y = iris.target





def XGB_cv(max_depth,learning_rate, n_estimators, gamma

           ,min_child_weight, max_delta_step, subsample

           ,colsample_bytree, silent=True, nthread=-1):

    model = xgb.XGBClassifier(max_depth=int(max_depth),

                              learning_rate=learning_rate,

                              n_estimators=int(n_estimators),

                              silent=silent,

                              nthread=nthread,

                              gamma=gamma,

                              min_child_weight=min_child_weight,

                              max_delta_step=max_delta_step,

                              subsample=subsample,

                              colsample_bytree=colsample_bytree)

    RMSE = cross_val_score(model, X, y, scoring='accuracy', cv=5).mean()

    return -RMSE



# 주어진 범위 사이에서 적절한 값을 찾는다.

pbounds = {'max_depth': (5, 10),

          'learning_rate': (0.01, 0.3),

          'n_estimators': (50, 1000),

          'gamma': (1., 0.01),

          'min_child_weight': (2, 10),

          'max_delta_step': (0, 0.1),

          'subsample': (0.7, 0.8),

          'colsample_bytree' :(0.5, 0.99)

          }



xgboostBO = BayesianOptimization(f = XGB_cv,pbounds = pbounds, verbose = 2, random_state = 1 )



# 메소드를 이용해 최대화!

xgboostBO.maximize(init_points=2, n_iter = 10)



xgboostBO.max # 찾은 파라미터 값 확인
fit_xgb = xgb.XGBClassifier(max_depth= int( xgboostBO.max['params']['max_depth'] ),

                             learning_rate=xgboostBO.max['params']['learning_rate'],

                             n_estimators=int(xgboostBO.max['params']['n_estimators']),

                             gamma= xgboostBO.max['params']['gamma'],

                             min_child_weight=xgboostBO.max['params']['min_child_weight'],

                             max_delta_step=xgboostBO.max['params']['max_delta_step'],

                             subsample=xgboostBO.max['params']['subsample'],

                             colsample_bytree=xgboostBO.max['params']['colsample_bytree'])

model  = fit_xgb.fit(X,y)