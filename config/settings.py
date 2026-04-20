from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_env: str = "simulation"
    refresh_interval_seconds: int = 120

    class Config:
        env_file = ".env"

settings = Settings()
