import json

import msgpack
import pytest
from falcon import testing
from pydantic import BaseModel

from ventu import Ventu


class Req(BaseModel):
    text: str


class Resp(BaseModel):
    spam: bool


class AntiSpam(Ventu):
    def inference(self, data):
        return True if '@' in data.text else False

    def postprocess(self, data):
        return {'spam': data}


@pytest.fixture(params=[True, False])
def client(request):
    ventu = AntiSpam(Req, Resp, use_msgpack=request.param)
    client = testing.TestClient(ventu.app)
    client.unpack = lambda x: msgpack.unpackb(x, raw=False) if request.param else json.loads(x)
    client.pack = lambda x: msgpack.packb(x) if request.param else json.dumps(x)
    return client


def test_http(client):
    # root
    resp = client.simulate_request('GET', '/')
    assert resp.status_code == 200
    assert client.unpack(resp.content) == {
        'health check': {'/health': 'GET'},
        'inference': {'/inference': 'POST'},
        'API document': {'/apidoc/redoc': 'GET', '/apidoc/swagger': 'GET'}
    }

    # health check
    resp = client.simulate_request('GET', '/health')
    assert resp.status_code == 200
    assert client.unpack(resp.content) == {
        'service': 'OK',
        'inference': 'OK',
        'preprocess': 'OK',
        'postprocess': 'OK',
    }

    # inference
    resp = client.simulate_request('GET', '/inference')
    assert resp.status_code == 405

    resp = client.simulate_request('POST', '/inference', body=client.pack({'str': 'text'}))
    assert resp.status_code == 422

    resp = client.simulate_request('POST', '/inference', body=client.pack({'text': '@'}))
    assert resp.status_code == 200
    assert client.unpack(resp.content) == {'spam': True}

    resp = client.simulate_request('POST', '/inference', body=client.pack({'text': 'hello'}))
    assert resp.status_code == 200
    assert client.unpack(resp.content) == {'spam': False}
