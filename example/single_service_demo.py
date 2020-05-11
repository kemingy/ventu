import logging
import pathlib
from typing import Tuple

import numpy
import onnxruntime
from pydantic import BaseModel

from ventu import Ventu


# define the input schema
class Input(BaseModel):
    text: Tuple[(str,) * 3]


# define the output schema
class Output(BaseModel):
    label: Tuple[(bool,) * 3]


class CustomModel(Ventu):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load model
        self.sess = onnxruntime.InferenceSession(model_path)
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

    model_path = pathlib.Path(__file__).absolute().parent / 'sigmoid.onnx'
    model = CustomModel(str(model_path), Input, Output)
    model.run_http(host='localhost', port=8000)

    """
    # try with `httpie`
    ## health check
        http :8000/health
    ## inference
        http POST :8000/inference text:='["hello", "world", "test"]'
    """
