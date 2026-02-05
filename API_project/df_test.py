import pandas as pd
import json
import numpy as np

df = pd.DataFrame(
    {
        "name": ["jessica", "Timothy"],
        "age": [3, 5],
        "hobbies": ["reading", "travelling"]
    }
)

transposed_df = df.T
dict_df = transposed_df.to_dict()
dict_df_values = transposed_df.to_dict().values()
dict_df_other = dict(transposed_df)
print("dict_df: ", dict_df)
print("dict_df_other: ", dict_df_other)
print("dict_df_values_only is: ", type(json.loads(json.dumps(list(dict_df_values))))[0])
print("new_way_conversion I is: ", type(np.array(df.to_dict(orient="records"))))
print("new_way_conversion II is:", type(transposed_df.to_dict(orient="records")))


json_dump = json.dumps(dict_df)
# json_dump_val = json.dumps(dict_df_values)
print(f"json dump value is : {json_dump}")
# print(f"json dump values' value is : {json_dump_val}")
json_load = json.loads(json_dump)
# print(f"json_loads value is : {json_load} is its type is {json_load}")

deploy = "DEPLOY"
print(deploy in ["DEPLOY", "CHASE", "PREDICT"])
print(df)
print("transposed_df val: ",transposed_df.to_json(orient="split"))
print("df val: ", df.to_json(orient="split"))