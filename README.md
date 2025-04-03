# Обработка данных и классификация

## **Задание 1**  

Дан CSV-файл sales_data.csv с данными о продажах (колонки: date, product_id, price, quantity, category).
1. Загрузить данные и выполнить предобработку:
	- Удалить строки с пропусками.
	- Добавить колонку total_sales = price * quantity.
	- Найти категорию товара с наибольшей выручкой за весь период.
    - Рассчитать среднее и стандартное отклонение quantity в каждой категории

2.	Подобрать параметры линейной функции y = a⋅x + b с помощью scipy.optimize.curve_fit.
3.	Визуализировать исходные данные и аппроксимацию (matplotlib).
4.	Рассчитать MSE между исходными и предсказанными значениями.

## **Задание 2**

Обучить модель для классификации на датасете MNIST (цифры):  

1. Разделить данные на обучающую и тестовую выборки в соотношении 80/20.  
2. Построить нейронную сеть с архитектурой:  
   - Входной слой: 4 нейрона  
   - Скрытый слой: 8 нейронов (ReLU)  
   - Выходной слой: 3 нейрона (Softmax)  
3. Обучить модель с оптимизатором Adam.  
4. Визуализировать графики изменения accuracy и loss.  
5. Рассчитать метрики (precision, recall, F1-score) на тестовой выборке.

## **Код и реализация**  

#### Задание № 1 - Работа с базой данных
```
import pandas as pd
import numpy as np

database = pd.read_csv("C:\\testovoe\\sales_data.csv")

database = database.dropna(axis=0, how='any', inplace=False)

database['total_sales'] = database['price'] * database['quantity']

many = (database.groupby('category')['price'].agg(['sum'])).idxmax()

middle_and_dismiss = database.groupby('category')['quantity'].agg(['mean', 'std'])


print(many)
print(database)
print(middle_and_dismiss)

```
#### Задание № 2 - Параметры для линейной функции
```
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def linear(a, x, b):
    return  a * x + b

x = np.array([0, 1, 2, 3, 4, 5])  
y = np.array([1.2, 2.7, 4.1, 5.5, 7.8, 9.1])  

mas, mat= curve_fit(linear,x,y)

a,b = mas
x_os = np.linspace(min(x), max(x), 6) 
y_os = linear(x_os, a, b)
```
##### средняя квадратичная ошибка
```
mse = np.mean((y_os - y)**2)
```
##### вывод
```
print(mse)
plt.scatter(x, y) 
plt.plot(x_os, y_os)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```
#### Задание № 3 - Классификация цифр
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```
##### Загрузка датасета
```
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
```

```
x = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

mask = np.isin(y, [0,1,2])
x = x[mask]
y = y[mask]

x = x.astype("float32") / 255
x = x.reshape(-1, 28*28)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
```
##### Архитектура модели
```
model = keras.Sequential([layers.Dense(4, activation='linear', input_shape=(784,)),
                         layers.Dense(8, activation='relu'),
                         layers.Dense(3, activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_test, y_test))
```
##### Вывод результата и графика
```

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train ')
plt.plot(history.history['val_accuracy'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)


print(classification_report(y_test, y_pred_classes, digits=3))
```


