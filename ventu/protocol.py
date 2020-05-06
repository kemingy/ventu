import logging
import socket
import struct
import json

import msgpack
from pydantic import ValidationError


class BatchProtocol:
    STRUCT_FORMAT = '!i'
    INT_BYTE_SIZE = 4
    INIT_MESSAGE = struct.pack(STRUCT_FORMAT, 0)

    def __init__(self, infer, req_schema, resp_schema, use_msgpack):
        self.req_schema = req_schema
        self.resp_schema = resp_schema
        self.use_msgpack = use_msgpack
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.packer = msgpack.Packer(
            autoreset=True,
            use_bin_type=True,
        )
        self.logger = logging.getLogger(__name__)
        self.infer = infer

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
        batch = msgpack.unpackb(self._request(conn), raw=False)
        ids = list(batch.keys())
        self.logger.debug(f'Received job ids: {ids}')

        # validate request
        validated = []
        errors = []
        for i, byte in enumerate(batch.values()):
            try:
                data = self._unpack(byte)
                obj = self.req_schema(**data if isinstance(data, dict) else data)
                validated.append(obj)
            except ValidationError as err:
                errors.append((i, self._pack(err.errors())))
                self.logger.info(
                    f'Job {ids[i]} validation error',
                    extra={'Validation': err.errors()}
                )
            except (json.JSONDecodeError,
                    msgpack.ExtraData, msgpack.UnpackValueError) as err:
                errors.append((i, self._pack(str(err))))
                self.logger.info(f'Job {ids[i]} error: {err}')

        # inference
        result = self.infer(validated)
        assert len(result) == len(validated), (
            'Wrong number of inference results. '
            f'Expcet {len(validated)}, get{len(result)}.'
        )

        # validate response
        for data in result:
            self.resp_schema.validate(data)

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

    def run(self, addr):
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
        self.logger.info('Close socket')
        self.sock.close()
