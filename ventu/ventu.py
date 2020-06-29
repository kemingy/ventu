import logging
from wsgiref import simple_server

from prometheus_client import Summary, Gauge, CollectorRegistry

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

        self.req_example = req_schema.Config.schema_extra.get('example')
        self.resp_example = resp_schema.Config.schema_extra.get('example')
        self.warmup_size = req_schema.Config.schema_extra.get('batch_size', 1)
        if self.resp_example:
            assert self.req_example, \
                'require request examples if response examples are provided'

        self._app = None
        self._sock = None
        self.config = Config(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.metric_registry = CollectorRegistry()
        self.SINGLE_PROCESS_TIME = Summary(
            'single_process_time',
            'Time spent in different part of the processing',
            ('process',),
            registry=self.metric_registry,
        )
        self.BATCH_PROCESS_TIME = Summary(
            'batch_process_time',
            'Time spent in different part of the processing',
            ('process',),
            registry=self.metric_registry,
        )
        self.WORKER = Gauge(
            'process_worker',
            'numbers of workers',
            ('protocol',),
            multiprocess_mode='livesum',
            registry=self.metric_registry,
        )

    def health_check(self, batch=False):
        """
        health check for model inference (can also be used to warm-up)

        :param bool batch: batch inference or single inference (default)
        :return bool: ``True`` if passed health check
        """
        if not self.req_example:
            self.logger.info('Please provide examples for inference warm-up')
            return

        if not batch:
            self.logger.info(f'Single inference warm-up with example: {self.req_example}')
            result = self._single_infer(self.req_schema.parse_obj(self.req_example))
            if self.resp_example:
                self.logger.info('Check single inference warm-up result')
                self.resp_schema.parse_obj(self.resp_example)
                assert self.resp_example == result, \
                    f'does not match {self.resp_example} != {result} for {self.req_example}'
        else:
            self.logger.info(f'Batch inference warm-up with size: {self.warmup_size}')
            examples = [
                self.req_schema.parse_obj(self.req_example) for _ in range(self.warmup_size)]
            results = self._batch_infer(examples)
            if self.resp_example:
                self.logger.info('Check batch inference warm-up results')
                for i in range(len(self.resp_example)):
                    self.resp_schema.parse_obj(results[i])
                    assert results[i] == self.resp_example[i], \
                        f'does not match {self.resp_example[i]} != {results[i]} for {examples[i]}'

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
                self.metric_registry,
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
        with self.WORKER.labels('http').track_inprogress():
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

    def run_unix(self, addr=None):
        """
        run as an inference worker with Unix domain socket

        :param string addr: socket file address
        """
        self.logger.info(f'Run Unix domain socket on {addr}')
        with self.WORKER.labels('unix').track_inprogress():
            self.sock.run(addr or self.config.socket, 'unix')

    def run_tcp(self, host=None, port=None):
        """
        run as an inference worker with TCP

        :param string host: host address
        :param int port: service port
        """
        host = host or self.config.host
        port = port or self.config.port
        self.logger.info(f'Run TCP service on {host}:{port}')
        with self.WORKER.labels('unix').track_inprogress():
            self.sock.run((host, port), 'tcp')

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
        with self.SINGLE_PROCESS_TIME.labels(process='preprocess').time():
            data = self.preprocess(data)
        with self.SINGLE_PROCESS_TIME.labels(process='inference').time():
            data = self.inference(data)
        with self.SINGLE_PROCESS_TIME.labels(process='postprocess').time():
            data = self.postprocess(data)
        return data

    def _batch_infer(self, batch):
        with self.BATCH_PROCESS_TIME.labels(process='preprocess').time():
            batch = [self.preprocess(data) for data in batch]
        with self.BATCH_PROCESS_TIME.labels(process='inference').time():
            batch = self.batch_inference(batch)
        with self.BATCH_PROCESS_TIME.labels(process='postprocess').time():
            batch = [self.postprocess(data) for data in batch]
        return batch
