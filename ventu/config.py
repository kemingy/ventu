from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """
    default config, can be rewrite with environment variables begin with ``ventu_``

    :ivar name: default service name shown in OpenAPI
    :ivar version: default service version shown in OpenAPI
    :ivar host: default host address for the HTTP service
    :ivar port: default port for the HTTP service
    :ivar socket: default socket file to communicate with batching service
    """
    name: str = Field('Deep Learning Service', env='name')
    version: str = Field('latest', env='version')
    host: str = Field('localhost', env='host')
    port: int = Field(8000, ge=80, le=65535, env='port')
    socket: str = Field('batching.socket', env='socket')

    class Config:
        env_prefix = 'ventu_'
