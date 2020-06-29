# Ventu

[![pypi](https://img.shields.io/pypi/v/ventu.svg)](https://pypi.python.org/pypi/ventu)
[![versions](https://img.shields.io/pypi/pyversions/ventu.svg)](https://github.com/zenchars/ventu)
![Python Test](https://github.com/kemingy/ventu/workflows/Python%20package/badge.svg)
[![Python document](https://github.com/kemingy/ventu/workflows/Python%20document/badge.svg)](https://kemingy.github.io/ventu)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/kemingy/ventu.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kemingy/ventu/context:python)

Serving the deep learning models easily.

## Install

```sh
pip install ventu
```

## Features

* only need to implement Model(`preprocess`, `postprocess`, `inference` or `batch_inference`)
* request & response data validation using [pydantic](https://pydantic-docs.helpmanual.io)
* API document using [SpecTree](https://github.com/0b01001001/spectree) (when run with `run_http`)
* backend service using [falcon](falcon.readthedocs.io/) supports both JSON and [msgpack](https://msgpack.org/)
* dynamic batching with [batching](https://github.com/kemingy/batching) using Unix domain socket or TCP
    * errors in one request won't affect others in the same batch
    * load balancing
* support all the runtime
* health check
* monitoring metrics (Prometheus)
    * if you have multiple workers, remember to setup `prometheus_multiproc_dir` environment variable to a directory
* inference warm-up

## How to use

* define your request data schema and response data schema with `pydantic`
    * add examples to `schema.Config.schema_extra[examples]` for warm-up and health check (optional)
* inherit `ventu.Ventu`, implement the `preprocess` and `postprocess` methods
* for standalone HTTP service, implement the `inference` method, run with `run_http`
* for the worker behind dynamic batching service, implement the `batch_inference` method, run with `run_socket`

check the [document](https://kemingy.github.io/ventu) for API details

## Example

The demo code can be found in [examples](examples).

### Service

Install requirements `pip install numpy torch transformers httpx`

```python
import argparse
import logging

import numpy as np
import torch
from pydantic import BaseModel, confloat, constr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from ventu import Ventu


# request schema used for validation
class Req(BaseModel):
    # the input sentence should be at least 2 characters
    text: constr(min_length=2)

    class Config:
        # examples used for health check and warm-up
        schema_extra = {
            'example': {'text': 'my cat is very cut'},
            'batch_size': 16,
        }


# response schema used for validation
class Resp(BaseModel):
    positive: confloat(ge=0, le=1)
    negative: confloat(ge=0, le=1)


class ModelInference(Ventu):
    def __init__(self, *args, **kwargs):
        # initialize super class with request & response schema, configs
        super().__init__(*args, **kwargs)
        # initialize model and other tools
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english')

    def preprocess(self, data: Req):
        # preprocess a request data (as defined in the request schema)
        tokens = self.tokenizer.encode(data.text, add_special_tokens=True)
        return tokens

    def batch_inference(self, data):
        # batch inference is used in `socket` mode
        data = [torch.tensor(token) for token in data]
        with torch.no_grad():
            result = self.model(torch.nn.utils.rnn.pad_sequence(data, batch_first=True))[0]
        return result.numpy()

    def inference(self, data):
        # inference is used in `http` mode
        with torch.no_grad():
            result = self.model(torch.tensor(data).unsqueeze(0))[0]
        return result.numpy()[0]

    def postprocess(self, data):
        # postprocess a response data (returned data as defined in the response schema)
        scores = (np.exp(data) / np.exp(data).sum(-1, keepdims=True)).tolist()
        return {'negative': scores[0], 'positive': scores[1]}


def create_model():
    logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    model = ModelInference(Req, Resp, use_msgpack=True)
    return model


def create_app():
    """for gunicorn"""
    return create_model().app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ventu service')
    parser.add_argument('--mode', '-m', default='http', choices=('http', 'unix', 'tcp'))
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', '-p', default=8080, type=int)
    parser.add_argument('--socket', '-s', default='batching.socket')
    args = parser.parse_args()

    model = create_model()
    if args.mode == 'unix':
        model.run_unix(args.socket)
    elif args.mode == 'tcp':
        model.run_tcp(args.host, args.port)
    else:
        model.run_http(args.host, args.port)
```

You can run this script as:

* a single thread HTTP service: `python examples/app.py`
* a HTTP service with multiple workers: `gunicorn -w 2 -b localhost:8080 'examples.app:create_app()'`
    * when run as a HTTP service, can check the follow links:
        * `/metrics` Prometheus metrics
        * `/health` health check
        * `/inference` inference
        * `/apidoc/redoc` or `/apidoc/swagger` OpenAPI document
* an inference worker behind the batching service: `python examples/app.py -m socket` (Unix domain socket) or `python examples/app.py -m tcp --host localhost --port 8888` (TCP) (need to run the [batching service](https://github.com/kemingy/batching) first)

### Client

```python
from concurrent import futures

import httpx
import msgpack

URL = 'http://localhost:8080/inference'
HEADER = {'Content-Type': 'application/msgpack'}
packer = msgpack.Packer(
    autoreset=True,
    use_bin_type=True,
)


def request(text):
    return httpx.post(URL, data=packer.pack({'text': text}), headers=HEADER)


if __name__ == "__main__":
    with futures.ThreadPoolExecutor() as executor:
        text = [
            'They are smart',
            'what is your problem?',
            'I hate that!',
            'x',
        ]
        results = executor.map(request, text)
        for i, resp in enumerate(results):
            print(
                f'>> {text[i]} -> [{resp.status_code}]\n'
                f'{msgpack.unpackb(resp.content)}'
            )
```
