# Ventu

[![pypi](https://img.shields.io/pypi/v/ventu.svg)](https://pypi.python.org/pypi/ventu)
[![versions](https://img.shields.io/pypi/pyversions/ventu.svg)](https://github.com/zenchars/ventu)

Serving the deep learning models easily.

## Install

```sh
pip install vento
```

## Features

* Only need to implement Model(`preprocess`, `postprocess`, `inference` or `batch_inference`)
* request & response data validation using [pydantic](https://pydantic-docs.helpmanual.io)
* API document using [SpecTree](https://github.com/0b01001001/spectree) (when run with `run_http`)
* backend service using [falcon](falcon.readthedocs.io/) supports both JSON and [msgpack](https://msgpack.org/)
* dynamic batching with [batching](https://github.com/kemingy/batching) using Unix Domain Socket
    * errors in one request won't affect others in the same batch
* support all the runtime
* health check

## How to use

* define your request data schema and response data schema with `pydantic`
* inherit `ventu.Ventu`, implement the `preprocess` and `postprocess` methods
* for standalone HTTP service, implement the `inference` method, run with `run_http`
* for the worker behind dynamic batching service, implement the `batch_inference` method, run with `run_socket`

check the [document](https://kemingy.github.io/ventu) for API details

## Example

### Dynamic Batching Demo

**Server**

Need to run the [batching](https://github.com/kemingy/batching) server first.

The demo code can be found in [batching demo](https://github.com/kemingy/batching/examples).

```python
import logging
from pydantic import BaseModel
from ventu import Ventu


class Req(BaseModel):
    num: int


class Resp(BaseModel):
    square: int


class ModelInference(Ventu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, data: Req):
        return data.num

    def batch_inference(self, data):
        return [num ** 2 for num in data]

    def postprocess(self, data):
        return {'square': data}


if __name__ == "__main__":
    logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    model = ModelInference(Req, Resp, use_msgpack=True)
    model.run_socket('batching.socket')
```

**Client**

```python
from concurrent import futures
import httpx
import msgpack


URL = 'http://localhost:8080'
packer = msgpack.Packer(
    autoreset=True,
    use_bin_type=True,
)


def request(text):
    return httpx.post(URL, data=packer.pack({'num': text}))


if __name__ == "__main__":
    with futures.ThreadPoolExecutor() as executor:
        text = (0, 'test', -1, 233)
        results = executor.map(request, text)
        for i, resp in enumerate(results):
            print(
                f'>> {text[i]} -> [{resp.status_code}]\n'
                f'{msgpack.unpackb(resp.content, raw=False)}'
            )
```

### Single Service Demo

source code can be found in [single_service_demo.py](example/single_service_demo.py)

```python
from ventu import Ventu
from typing import Tuple
from pydantic import BaseModel
import logging
import numpy
import onnxruntime


# define the input schema
class Input(BaseModel):
    text: Tuple[(str,) * 3]


# define the output schema
class Output(BaseModel):
    label: Tuple[(bool,) * 3]


class CustomModel(Ventu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load model
        self.sess = onnxruntime.InferenceSession('./sigmoid.onnx')
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def preprocess(self, data: Input):
        # data format is defined in ``Input``
        words = [sent.split(' ')[:4] for sent in data.text]
        # padding
        words = [word + [''] * (4 - len(word)) for word in words]
        # build embedding
        emb = [[
            numpy.random.random(5) if w else [0] * 5
            for w in word]
            for word in words]
        return numpy.array(emb, dtype=numpy.float32)

    def inference(self, data):
        # model inference
        return self.sess.run([self.output_name], {self.input_name: data})[0]

    def postprocess(self, data):
        # generate the same format as defined in ``Output``
        return {'label': [bool(numpy.mean(d) > 0.5) for d in data]}


if __name__ == "__main__":
    logger = logging.getLogger()
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    model = CustomModel(Input, Output)
    model.run_http(host='localhost', port=8000)
```

try with `httpie`

```shell script
# health check
http :8000/health
# inference
http POST :8000/inference text:='["hello", "world", "test"]'
```

Open `localhost:8000/apidoc/redoc` in your browser to see the API document.

**Run with Gunicorn**

```shell script
gunicorn -w 2 model.app
```
