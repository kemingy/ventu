import logging
import socket
import struct
import json

import msgpack


class BatchProtocol:
    STRUCT_FORMAT = '!i'
    INT_BYTE_SIZE = 4
    INIT_MESSAGE = struct.pack(STRUCT_FORMAT, 0)

    def __init__(self, infer, use_msgpack):
        self.use_msgpack = use_msgpack
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.packer = msgpack.Packer(
            autoreset=True,
            use_bin_type=True,
        )
        self.logger = logging.getLogger(__name__)
        self.infer = infer

    def unpack(self, data):
        return msgpack.unpackb(data) if self.use_msgpack else json.loads(data)

    def request(self, conn):
        self.logger.info('Send init message')
        conn.sendall(self.INIT_MESSAGE)

    def process(self, conn):
        length_bytes = conn.recv(self.INT_BYTE_SIZE)
        length = struct.unpack(self.STRUCT_FORMAT, length_bytes)[0]
        data = conn.recv(length)
        return data

    def response(self, conn, data):
        data = self.packer.pack(data)
        conn.sendall(struct.pack(self.STRUCT_FORMAT, len(data)))
        conn.sendall(data)

    def inference(self, data):
        jobs = msgpack.unpackb(data)
        result = self.infer([self.unpack(data) for data in jobs.values()])
        return dict(zip(jobs.keys(), result))

    def run(self, addr):
        self.logger.info(f'Connect to socket: {addr}')
        while True:
            try:
                self.sock.connect(addr)
                self.logger.info(f'Connect to {self.sock.getpeername()}')
                self.request(self.sock)

                while True:
                    data = self.process(self.sock)
                    data = self.inference(data)
                    self.response(self.sock, data)

            except BrokenPipeError as err:
                self.logger.warning(f'Broken socket: {err}')
                continue

    def stop(self):
        self.logger.info('Close socket')
        self.sock.close()
