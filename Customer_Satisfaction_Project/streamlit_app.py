import json

import numpy as np
import pandas as pd
import mlflow
import streamlit as st
from PIL import Image

from pipelines.deployment_pipeline import prediction_service_loader
from pipelines.utils import get_data_for_test
from run_deployment_prog import run_deployment

with open("model_deployment.json", "r") as f:
    deployment_info = json.load(f)

str_data = get_data_for_test()
model_uri = deployment_info["model_uri"]
model = mlflow.sklearn.load_model(model_uri)

def restructure_df(
    # model_uri: str,
    data: str
) -> pd.DataFrame:
    print(f"Loading model from: {model_uri}")
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    desired_columns = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_length",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    df = pd.DataFrame(data["data"])
    df = df.T
    df.columns = desired_columns

    print("df value in restructured_df func is:", df)
    return df

    # df = pd.DataFrame(data["data"])
    # df = df.T
    # df.columns = desired_columns
    # prediction = model.predict(df)
    # print(f"predictions: {prediction}")

def main():
    st.title("End to End Customer Satisfaction Pipeline with Zenml.")

    high_level_image = Image.open("_assets/high_level_overview.png")
    st.image(high_level_image, caption="High Level Pipeline")

    whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    st.markdown(
        """
    ### Problem Statement
    The objective here is  to predict the customer satisfaction score for a given order based on specific features in the dataset
        """
    )

    st.image(whole_pipeline_image, caption="whole pipeline")
    st.markdown(
        """
    Above is a figure of the whole pipeline, we first ingest, then clean the data,train a model the work on the data, and evaluate the model's performance 
        """
    )
    st.markdown(
    """
    #### Description of features
    This app is designed to predict the customers satisfaction score for a given customer
    | Models          | Description             |
    """
    )
    if 'input' not in st.session_state:
        st.session_state.input = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("manual input"):
            st.session_state.input = "manual"

    with col2:
        if st.button("generate input"):
            st.session_state.input = "generate"
    
    # payment_sequential = st.sidebar.slider("Payment Sequential")
    # payment_installments = st.sidebar.slider("Payment Installments")
    # payment_value = st.number_input("Payment_value")
    # price = st.number_input("Price")
    # freight_value = st.number_input("freight_value")
    # product_name_length = st.number_input("Product name length")
    # product_description_length = st.number_input("Product Description length")
    # product_photos_qty = st.number_input("Product Photos Quantity")
    # product_weigh_g = st.number_input("Product weight Measured in grams")
    # product_length_cm = st.number_input("Product lenght (CMs)")
    # product_height_cm = st.number_input("Product height (CMs)")
    # product_width_cm = st.number_input("Product width (CMs)")

    df = None
    st.markdown("Would you like to manually input data or dynamically generate data?")
    if st.session_state.input == "manual":
            # st.success(
            #     f"Your customer Satisfactory rate(range between 0 - 5) with given product details:\
            #         {prediction}"
            # )
            payment_sequential = st.sidebar.slider("Payment Sequential")
            payment_installments = st.sidebar.slider("Payment Installments")
            payment_value = st.number_input("Payment_value")
            price = st.number_input("Price")
            freight_value = st.number_input("freight_value")
            product_name_length = st.number_input("Product name length")
            product_description_length = st.number_input("Product Description length")
            product_photos_qty = st.number_input("Product Photos Quantity")
            product_weigh_g = st.number_input("Product weight Measured in grams")
            product_length_cm = st.number_input("Product lenght (CMs)")
            product_height_cm = st.number_input("Product height (CMs)")
            product_width_cm = st.number_input("Product width (CMs)")

            df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weigh_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm]
                }
            )
    # if st.button("generate inputi"):
    if st.session_state.input == "generate":
        n_df = restructure_df(data=str_data)
        df = pd.DataFrame(
        {
            "payment_sequential": n_df["payment_sequential"].values,
            "payment_installments": n_df["payment_installments"].values,
            "payment_value": n_df["payment_value"].values,
            "price": n_df["price"].values,
            "freight_value": n_df["freight_value"].values,
            "product_name_lenght": n_df["product_name_length"].values,
            "product_description_lenght": n_df["product_description_lenght"].values,
            "product_photos_qty": n_df["product_photos_qty"].values,
            "product_weight_g": n_df["product_weight_g"].values,
            "product_length_cm": n_df["product_length_cm"].values,
            "product_height_cm":n_df["product_height_cm"].values,
            "product_width_cm": n_df["product_width_cm"].values
            }
        )

        # st.write(f" i am using the write method to view the output here: {st.dataframe(gen_df)}")

        st.dataframe(df)

        # print(f"testing for the content of payment_sequential val: {n_df["payment_sequential"].values}")
        # print(f"testing for the content of payment_installments val: {n_df["payment_installments"]}")

        
    # json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    # json_list = json.to_dict(orient="records")
    # transposed_df = df.T
    # data = np.array(json_l)
        # print("done with processing")
        # print(n_df)
        # print("X" * 400)
        # print(gen_df)
    if df is not None and  st.button("Predict"):
        # service = prediction_service_loader(
        #     pipeline_name="continuous_deployment_pipeline",
        #     pipeline_step_name="deploy_model",
        #     running=False,
        # )
        # if service is None:
        #     st.write(
        #         "No service found. The pipeline will be run first to create the service"
        #     )
        #     run_deployment()
        
        

        prediction = model.predict(df)
        st.success(
            f"Your customer Satisfactory rate(range between 0 - 5) with given product details:\
                {prediction}"
        )

    # if st.button("Results"):
    #     st.write(
    #         "We have experimented with two ensemble and tree based models and compared the"
    #     )

    # df = pd.DataFrame(
    #     {
    #         "Models": ["LightGBM", "Xgboost"],
    #         "MSE": [1.804, 1.781],
    #         "RMSE": [1.343, 1.335]
    #     }
    # )

    # st.dataframe(df)

    # st.write(
    #     "Following figure shows how important each feature is in the model that contributes to its overall performance"
    # )
    # image = Image.open("_assets/feature_importance_gain.png")
    # st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()
