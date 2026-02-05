import json
import pandas as pd
import numpy as np

from pipelines.utils import get_data_for_test


# a = np.array([[],[]])
# print(a.shape)
# df = pd.read_csv("./dataset/olist_customers_dataset.csv")
dat = get_data_for_test()

# print(df)
dat = json.loads(dat)
dat.pop("columns")
dat.pop("index")
desired_columns = [
    "payment_sequential",
    "payment_installments",
    "payment_values",
    "price",
    "freight_value",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

val = int(input("pick a value"))

if val < 3:
    print("This value is greater than 3")
    return val * 3
if val == 3:
    print("This value is equal to 3")
    return 3 * 3
else:
    return -1



print()
