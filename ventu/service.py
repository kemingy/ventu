from wsgiref import simple_server

from .model import VentuModel
from .config import Config
from .handlers import create_app


class VentuService:
    def __init__(self, model_cls, json_schema, resp_schema, *args, **kwargs):
        self.config = Config()
        assert issubclass(model_cls, VentuModel), (
            "The model doesn't implement all the class method: "
            "(`inference`, `preprocess`, `postprocess`)"
        )
        self.app = create_app(
            model_cls(*args, **kwargs, **self.config.dict()),
            json_schema,
            resp_schema,
            self.config,
        )

    def run(self, host=None, port=None, **kwargs):
        """
        :param str host: service host
        :param int port: service port
        :param kwargs: ``gunicorn`` configs
        """
        httpd = simple_server.make_server(
            host or self.config.host,
            port or self.config.port,
            self.app
        )
        httpd.serve_forever()
