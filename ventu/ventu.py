from wsgiref import simple_server

from .handlers import create_app
from .config import Config


class Ventu:
    def __init__(self, req_schema, resp_schema, use_msgpack=False, *args, **kwargs):
        self.req_schema = req_schema
        self.resp_schema = resp_schema
        self.use_msgpack = use_msgpack
        self._app = None
        self._sock = None
        self.config = Config()

    @property
    def app(self):
        """
        Falcon application with SpecTree validation
        """
        if self._app is None:
            self._app = create_app(
                self, self.req_schema, self.resp_schema, self.use_msgpack, self.config
            )
        return self._app

    def run_http(self, host, port):
        httpd = simple_server.make_server(
            host or self.config.host,
            port or self.config.port,
            self.app
        )
        httpd.serve_forever()

    @property
    def sock(self):
        if self._sock is None:
            self._sock = create_sock()
        return self._sock

    def run_socket(self, addr):
        pass

    def inference(self, data):
        return data

    def preprocess(self, data):
        return data

    def postprocess(self, data):
        return data

    def _infer(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data
