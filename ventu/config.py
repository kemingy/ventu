from pydantic import BaseSettings, Field


class Config(BaseSettings):
    name: str = Field('Deep Learning Service', env='name')
    version: str = Field('latest', env='version')
    host: str = Field('localhost', env='host')
    port: int = Field(8000, ge=80, le=65535, env='port')

    class Config:
        env_prefix = 'ventu_'
