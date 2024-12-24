# Вопросы

1. Линейная регрессия

    - Классическая модель, спецификация (соответствие данных)
    - МНК
    - Тестирование гипотез и построение доверительных интервалов
    - Проблема мультиколлинеаронсти
    - Фиктивные переменные
    - Гетероскедатичность и обобщенная линейная модель

2. Панельные данные
3. Прогнозирование на основе регрессии
4. Деревья классификации и регрессии
5. Случайный лес
6. Виды градиентного бустинга. Ключевые параметры модели

    - Ансамблевые модели (бустинг над деревьями решений, виды, кросс-валидация)

7. Временные ряды

    - AR, MA, ARMA, ARIMA, GARCH, ... моделирование волатильности

8. Модели оценки рисков и доходностей в инвестициях

    - оптимальный портфель

# Штуки

- [ ] статистика
- [ ] эконометрика

- [x] линейная регрессия
- [ ] парная регрессия
- [ ] метод наименьших квадратов
- [ ] мультиколлинеарность

- [ ] критерий Фишера (F-статистика)
- [ ] коэффициент корелляции пирсона
- [ ] матрица корелляции
- [ ] variance inflation factor
- [ ] коэффициент детерминации R2

- [ ] панельные данные
- [ ] эндогенность
- [ ] long/wide panel dataset
- [x] сбалансированность / несбалансированность
- [ ] гомогенномть / гетерогенность
- [ ] смещение данных
- [ ] истощение выборки

- [ ] Функция связи
- [ ] GLM (обобщенные лиейные модели)
- [ ] Биномиальная регрессия
- [ ] Логистическая регрессия
- [ ] Логистическая функция
- [ ] Probit
- [ ] Мультиномиальная регрессия
- idk there is more

- [ ] метрики классификации

    - [x] матрица ошибок (confusion matrix)
    - [ ] accuracy / error rate
    - [ ] precision / recall (tpr) + AUC
    - [ ] TRN / FPR
    - [ ] F1-score
    - [ ] ROC-AUC

TODO Trees and forests

- [ ] ансамблевые методы

    - [x] бэггинг
    - [x] бустинг
    - [ ] стэкинг

- [ ] Подбор гиперпараметров

    - [ ] grid search
    - [ ] random search
    - [ ] байесовская оптимизация

- [x] темп обучения (learning rate)
- [x] кросс валидация

- [ ] временной ряд, уровень ряда, длина ряда
- [ ] классификация временных рядов
- [ ] аддитивная и мультипликативная модель
- [ ] типы трендов
- [ ] предсказательный интервал
- [ ] авторегрессия (AR)
- [ ] скользящее среднее
- [ ] интегрированные процессы

# Linear Regression

Линейная регрессия - регрессия линейной зависимости.
То есть ответ является линейной комбинацией признаков.
Обучение состоит в нахождении коэффициентов линейной комбинации.

Достоинства:

- Простота в реализации и интерпретации
- Высокая скорость работы
- Хорошая точность в случае линейной зависимости

Недостатки:

- Не работает с нелинейными зависимостями
- Чувствительность к выбросам
- Предполагает независимость, гомоскедатичность и нормальность

В sklearn есть МНК, который называется LinearRegression.
Почти никаких параметров нет.

В sklearn есть еще линейные регрессии Ridge, Lasso, ElasticNet.

Они используются для того чтобы предотвратить переобучение
Они используют регуляризацию.

Регуляризация - метод добавления некоторых дополнительных ограничений.
Так называемый штраф за сложность модели.

l2 ridge - squared parameters
l1 lasso - absolute parameters
elastictet - какая то комбинация l1 и l2 и еще какого то слагаемого

# General Linear Model

Same as linear model but with matrices and not vectors
Если штуки будут столбйовыми векторами, то получится multiple linear model

# Generalized Linear Model

It is a different thing

# Logistic Regression

В sklearn модель называется LogisticRegression

LogisticRegression parameters:
```
penalty = l2 (l1, l2, elasticnet) - the norm of the penalty
tol = 1e-4 - tolerance for stopping criteria
C = 1 - the inverse of regularization strength
max_iter = 100
solver = lbfgs (lbfgs, liblinear, newton-cg, newton-cholesky, sag, saga)
intercept_scaling
class_weight
```

# Decision Trees

При построении дерева мы смотрим на все возможные разбиения листа и вбираем из них самое лучшее.
Например с минимальным SSR или с наименьшим gini impurity.

Достоинства:

- Простота
- Интерпретируемость
- Способность выявлять нелинейные зависимости

Недостатки:

- Склонность к переобучению
- Неумение экстраполировать

Мы можем делать прунинг (срижку) деревьев чтобы получить результат лучше.

Строим переобученное дерево с чистыми листьями на всех данных.
tree score там дада
строим кучу деревьев и выбираем там лучшее хз пофиг

# Random Forest

- Bagging.
- Out of bag.
- Mean decrease in impurity.

Достоинства:

- Высокая точность предсказаний
- Нечувствительность к выбросам
- Параллелизуемость
- Невозможность переобучения
- Простота реализации

Недостатки:

- Неумение экстраполировать (СЛ не может выдать ранее невиданное значение)
- Плохо работает с разреженными признаками
- Большой размер модели, требует много памяти


sklearn RandomForestClassifier parameters:
```
n_estimators=100,
criterion='gini' (gini, entropy, log_loss),
max_depth=None,
min_samples_split=2,
min_samples_leaf=1,
min_impurity_decrease=0
max_features='sqrt' (sqrt, log2, any number),
max_leaf_nodes=None,
bootstrap=True
oob_score=False
ccp_alpha=0
```

sklearn RandomForestRegressor parameters:
```
n_estimators=100,
criterion='squared_error' (squared_error, absolute_error, friedman_mse, poisson),
max_depth=None,
min_samples_split=2,
min_samples_leaf=1,
min_impurity_decrease=0
max_features=1.0 (sqrt, log2, any number),
max_leaf_nodes=None,
bootstrap=True
oob_score=False
ccp_alpha=0
```

# Gradient Boosting

Обычно ансамблируют именно деревья, но никто не запрещает ансамблировать другие модели.
Но вот только ансамбль линейных моделей будет линейным.

Each new tree is predicting residuals of previous model and contributes the prediction scaled by the learning rate to the current model

Достоинства:

- Эффективно находит нелинейные зависимости и несмотря на то что таким свойством имеют все модели осноманные на деревьях, GBDT выигрывает в большинстве случаев.

Недостатки:

- Переобучение

Max depth is to be tuned as sklearn suggests and by default it is 3

sklearn uses trees for boosting

sklearn GradientBoostingClassifier parameters:

```
loss=log_loss (log_loss(same as logistic regression), exponential(recovers AdaBoost))
learning_rate=0.1
n_estimators=100
subsample=1.0
criterion=friedman_mse (friedman_mse, squared_error) friedman is generally best
min_samples_split=2
min_samples_leaf=1,
min_weight_fraction_leaf=0
max_depth=3
min_impurity_decrease=0
max_features=None
validation_fractioin=0.1
n_iter_no_change=None
tol=1e-4
ccp_alpha=0
```

sklearn GradientBoostingRegressor parameters:

```
loss=squared_error (squared_error, absolute_error, huber, quantile)
learning_rate=0.1
n_estimators=100
subsample=1.0
criterion=friedman_mse (friedman_mse, squared_error) friedman is generally best
min_samples_split=2
min_samples_leaf=1,
min_weight_fraction_leaf=0
max_depth=3
min_impurity_decrease=0
max_features=None
validation_fractioin=0.1
n_iter_no_change=None
tol=1e-4
ccp_alpha=0
```

# Time Series

sklearn has TimeSeriesSplit

It is a time series cross-validator.

TimeSeriesSplit parameters:
```
n_splits=5
max_train_size=None
test_size=None
gap=0
```

Для рядов есть операторное представление and I don't care about it.
Можно делать всякое сглаживание.

Ряд нестационарный когда вероятностные свойства временного ряда изменяются во времени.

## Авторегрессия (AR, autoregression)

Авторегрессионная модель - значения мременного ряда в данны момент представимы как линейная комбинация конечного количества предыдущих значений этого ряда

## Скользящее среднее (MA, moving average)

Значение временного ряда - линейная комбинация шума всех предыдущих значений.

## ARMA

Модель авторегрессии - скользящего среднего - сумма AR и MA моделей.

То есть объясняющие переменные это прошлые значения зависимой переменной, ф в качестве регрессионного остатка - скользящее среднее из элементов белого шума

## ARIMA

Autoregressive integrated moving average

Обобщение на нестационарные ряды.

В нем в AR добавляется оператор разности временного ряда порядка d

## SARIMA

Seasonal ARIMA

Обобщение на периодичные ряды.

## ARIMAX

ARIMA extended

Добавляет возможность учета внешних факторов
