    data_df = pd.read_csv(r"\my_projects\customer_satisfaction_project\dataset\olist_customers_dataset.csv")
    data_clean_test = DataPreprocessStrategy().handle_data(data_df)
    print(data_clean_test.columns)
    data_split_test = DataSplitStrategy().handle_data(data_clean_test)
    pri