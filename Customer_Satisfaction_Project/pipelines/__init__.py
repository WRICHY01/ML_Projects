import pandas as pd

df = pd.read_csv(r"\my_projects\customer_satisfaction_project\dataset\olist_customers_dataset.csv")

print(df["review_score"].unique())