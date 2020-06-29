import json
import logging
import socket
import struct

import msgpack
from pydantic import ValidationError


class BatchProtocol:
    """
    protocol used to communicate with batching service

    :param infer: model infer function (contains `preprocess`, `batch_inference` and `postprocess`)
    :param req_schema: request schema defined with `pydantic`
    :param resp_schema: response schema defined with `pydantic`
    :param bool use_msgpack: use msgpack for serialization or not (default: JSON)
    """
    STRUCT_FORMAT = '!i'
    INT_BYTE_SIZE = 4
    INIT_MESSAGE = struct.pack(STRUCT_FORMAT, 0)

    def __init__(self, infer, req_schema, resp_schema, use_msgpack):
        self.req_schema = req_schema
        self.resp_schema = resp_schema
        self.use_msgpack = use_msgpack
        self.packer = msgpack.Packer(autoreset=True, use_bin_type=True)
        self.logger = logging.getLogger(__name__)
        self.infer = infer
        self.sock = None

    def _pack(self, data):
        return self.packer.pack(data) if self.use_msgpack else json.dumps(data)

    def _unpack(self, data):
        return msgpack.unpackb(data, raw=False) if self.use_msgpack else json.loads(data)

    def _init_request(self, conn):
        self.logger.info('Send init message')
        conn.sendall(self.INIT_MESSAGE)

    def _request(self, conn):
        length_bytes = conn.recv(self.INT_BYTE_SIZE)
        length = struct.unpack(self.STRUCT_FORMAT, length_bytes)[0]
        data = conn.recv(length)
        return data

    def process(self, conn):
        """
        process batch queries and return the inference results

        :param conn: socket connection
        """
        batch = msgpack.unpackb(self._request(conn), raw=False)
        ids = list(batch.keys())
        self.logger.debug(f'Received job ids: {ids}')

        # validate request
        validated = []
        errors = []
        for i, byte in enumerate(batch.values()):
            try:
                data = self._unpack(byte)
                obj = self.req_schema.parse_obj(data)
                validated.append(obj)
                self.logger.debug(f'{obj} passes the validation')
            except ValidationError as err:
                errors.append((i, self._pack(err.errors())))
                self.logger.info(
                    f'Job {ids[i]} validation error',
                    extra={'Validation': err.errors()}
                )
            except (json.JSONDecodeError,
                    msgpack.ExtraData, msgpack.FormatError, msgpack.StackError) as err:
                errors.append((i, self._pack(str(err))))
                self.logger.info(f'Job {ids[i]} error: {err}')

        # inference
        self.logger.debug(f'Validated: {validated}, Errors: {errors}')
        result = []
        if validated:
            result = self.infer(validated)
            assert len(result) == len(validated), (
                'Wrong number of inference results. '
                f'Expcet {len(validated)}, get{len(result)}.'
            )

        # validate response
        for data in result:
            self.resp_schema.parse_obj(data)

        # add errors information
        err_ids = ''
        result = [self._pack(data) for data in result]
        for index, err_msg in errors:
            err_ids += ids[index]
            result.insert(index, err_msg)

        # build batch job table
        resp = dict(zip(ids, result))
        if err_ids:
            resp['error_ids'] = err_ids
        self._response(conn, resp)

    def _response(self, conn, data):
        data = self.packer.pack(data)
        conn.sendall(struct.pack(self.STRUCT_FORMAT, len(data)))
        conn.sendall(data)

    def run(self, addr, protocol='unix'):
        """
        run socket communication

        this should run **after** the socket file is created by the batching service

        :param string protocol: 'unix' or 'tcp'
        :param addr: socket file path or (host:str, port:int)
        """
        self.sock = socket.socket(
            socket.AF_UNIX if protocol == 'unix' else socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self.logger.info(f'Connect to socket: {addr}')
        while True:
            try:
                self.sock.connect(addr)
                self.logger.info(f'Connect to {self.sock.getpeername()}')
                self._init_request(self.sock)

                while True:
                    self.process(self.sock)

            except BrokenPipeError as err:
                self.logger.warning(f'Broken socket: {err}')
                continue

    def stop(self):
        """
        stop the socket communication
        """
        self.logger.info('Close socket')
        self.sock.close()
