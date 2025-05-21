from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """
    Model configs.
    """
    model_name: str = "LinearRegression"

    

if __name__ == "__main__":
    config = ModelNameConfig()
    print(config.model_name)