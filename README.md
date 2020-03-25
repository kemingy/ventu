# Ventu

Help you host your deep learning models easily.

## Features

* Only need to implement Model(`inference`, `preprocess`, `postprocess`)
* request & response data check using [pydantic](https://pydantic-docs.helpmanual.io)
* API document using [SpecTree](https://github.com/0b01001001/spectree)
* health check

## Example

source code can be found in [demo.py](./example/demo.py)

```py
from ventu import VentuModel, VentuService
from typing import Tuple
from pydantic import BaseModel
import numpy
import onnxruntime


# define the input schema
class Input(BaseModel):
    x: Tuple[(str,) * 3]


# define the output schema
class Output(BaseModel):
    label: Tuple[(bool,) * 3]


class CustomModel(VentuModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # load model
        self.sess = onnxruntime.InferenceSession('./sigmoid.onnx')
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def preprocess(self, data: Input):
        # data format is defined in ``Input``
        words = [sent.split(' ')[:4] for sent in data.x]
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

    def postprocess(self, data) -> Output:
        # generate the same format as defined in ``Output``
        return {'label': [bool(numpy.mean(d) > 0.5) for d in data]}


if __name__ == "__main__":
    service = VentuService(CustomModel, Input, Output)
    service.run(host='localhost', port=8000)

```

## Run with Gunicorn

```sh
gunicorn -w 2 service.app
```
