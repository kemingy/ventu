import logging
import random
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
        self.req_examples = req_schema.Config.schema_extra.get('examples')
        self.resp_examples = resp_schema.Config.schema_extra.get('examples')
        if self.resp_examples:
            assert self.req_examples, \
                'require request examples if response examples are provided'
            assert len(self.req_examples) == len(self.resp_examples), \
                'cannot find corresponding examples'
        self._app = None
        self._sock = None
        self.config = Config()
        self.logger = logging.getLogger(__name__)

    def health_check(self, batch=False):
        """
        health check for model inference (can also be used to warm-up)

        :param bool batch: batch inference or single inference (default)
        :return bool: ``True`` if passed health check
        """
        if not self.req_examples:
            self.logger.info('Please provide examples for inference warm-up')
            return

        if not batch:
            index = random.choice(range(len(self.req_examples)))
            example = self.req_examples[index]
            self.logger.info(f'Single inference warm-up with example: {example}')
            result = self._single_infer(self.req_schema.parse_obj(example))
            if self.resp_examples:
                self.logger.info('Check single inference warm-up result')
                expect = self.resp_examples[index]
                self.resp_schema.parse_obj(expect)
                assert expect == result, \
                    f'does not match {expect} != {result} for {example}'
        else:
            self.logger.info('Batch inference warm-up')
            examples = [self.req_schema.parse_obj(data) for data in self.req_examples]
            results = self._batch_infer(examples)
            if self.resp_examples:
                self.logger.info('Check batch inference warm-up results')
                for i in range(len(self.resp_examples)):
                    self.resp_schema.parse_obj(results[i])
                    assert results[i] == self.resp_examples[i], \
                        f'does not match {self.resp_examples[i]} != {results[i]} for {examples[i]}'

        return True

    @property
    def app(self):
        """
        Falcon application with SpecTree validation
        """
        if self._app is None:
            self.health_check()
            self.logger.debug('Create Falcon application')
            self._app = create_app(
                self._single_infer,
                self.health_check,
                self.req_schema,
                self.resp_schema,
                self.use_msgpack,
                self.config,
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
            self.health_check(batch=True)
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
