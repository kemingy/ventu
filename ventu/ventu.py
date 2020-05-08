import logging
from wsgiref import simple_server

from .config import Config
from .protocol import BatchProtocol
from .service import create_app


class Ventu:
    """
    Ventu: built for deep learning model serving

    :param req_schema: request schema defined with :class:`pydantic.BaseModel`
    :param resp_schema: response schema defined with :class:`pydantic.BaseModel`
    :param bool use_msgpack: use msgpack for serialization or not (default: JSON)
    :param args:
    :param kwargs:

    To create a model service, inherit this class and implement:

        * ``preprocess`` (optional)
        * ``postprocess`` (optional)
        * ``inference`` (for standalone HTTP service)
        * ``batch_inference`` (when working with batching service)
    """

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
        """
        run the HTTP service

        :param string host: host address
        :param int port: service port
        """
        self.logger.info(f'Run HTTP service on {host}:{port}')
        httpd = simple_server.make_server(
            host or self.config.host,
            port or self.config.port,
            self.app
        )
        httpd.serve_forever()

    @property
    def sock(self):
        """
        socket used for communication with batching service

        this is a instance of :class:`ventu.protocol.BatchProtocol`
        """
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
        """
        run as an inference worker

        :param string addr: socket file address
        """
        self.logger.info(f'Run socket on {addr}')
        self.sock.run(addr or self.config.socket)

    def batch_inference(self, batch):
        """
        batch inference the preprocessed data

        :param batch: a list of data after :py:meth:`preprocess <ventu.ventu.Ventu.preprocess>`
        :return: a list of inference results
        """
        return batch

    def inference(self, data):
        """
        inference the preprocessed data

        :param data: data after :py:meth:`preprocess <ventu.ventu.Ventu.preprocess>`
        :return: inference result
        """
        return data

    def preprocess(self, data):
        """
        preprocess the data

        :param data: as defined in ``req_schema``
        :return: this will be the input data of
            :py:meth:`inference <ventu.ventu.Ventu.inference>`
            or one item of the input data of
            :py:meth:`batch_inference <ventu.ventu.Ventu.batch_inference>`
        """
        return data

    def postprocess(self, data):
        """
        postprocess the inference result

        :param data: data after :py:meth:`inference <ventu.ventu.Ventu.inference>`
            or one item of the :py:meth:`batch_inference <ventu.ventu.Ventu.batch_inference>`
        :return: as defined in ``resp_schema``
        """
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
