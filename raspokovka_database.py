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

