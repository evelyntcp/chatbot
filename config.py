import pydantic_settings

class Settings(pydantic_settings.BaseSettings):
    """
    Configuration class for application settings

    Attributes:
        API_NAME (str): The name of the API. Default is "chat".
        API_V1_STR (str): The base URL path for version 1 of the API. Default is "/api/v1".
    """

    API_NAME: str = "chat"
    API_V1_STR: str = "/api/v1"


SETTINGS = Settings()
