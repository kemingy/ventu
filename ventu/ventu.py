import logging
from wsgiref import simple_server

from .service import create_app
from .protocol import BatchProtocol
from .config import Config


class Ventu:
    def __init__(self, req_schema, resp_schema, use_msgpack=False, *args, **kwargs):
        self.req_schema = req_schema
        self.resp_schema = resp_schema
        self.use_msgpack = use_msgpack
        self._app = None
        self._sock = None
        self.config = Config()
        self.logger = logging.getLogger(__name__)

    @property
    def app(self):
        """
        Falcon application with SpecTree validation
        """
        if self._app is None:
            self.logger.debug('Create Falcon application')
            self._app = create_app(
                self._single_infer,
                self.req_schema,
                self.resp_schema,
                self.use_msgpack,
                self.config
            )
        return self._app

    def run_http(self, host=None, port=None):
        self.logger.info(f'Run HTTP service on {host}:{port}')
        httpd = simple_server.make_server(
            host or self.config.host,
            port or self.config.port,
            self.app
        )
        httpd.serve_forever()

    @property
    def sock(self):
        if self._sock is None:
            self.logger.debug('Create socket')
            self._sock = BatchProtocol(
                self._batch_infer,
                self.req_schema,
                self.resp_schema,
                self.use_msgpack,
            )
        return self._sock

    def run_socket(self, addr=None):
        self.logger.info(f'Run socket on {addr}')
        self.sock.run(addr or self.config.socket)

    def batch_inference(self, batch):
        return batch

    def inference(self, data):
        return data

    def preprocess(self, data):
        return data

    def postprocess(self, data):
        return data

    def _single_infer(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data

    def _batch_infer(self, batch):
        batch = [self.preprocess(data) for data in batch]
        batch = self.batch_inference(batch)
        batch = [self.postprocess(data) for data in batch]
        return batch
