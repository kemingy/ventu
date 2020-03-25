from enum import Enum
import logging
import falcon
from spectree import SpecTree, Response
from pydantic import BaseModel


class StatusEnum(str, Enum):
    ok = 'OK'
    error = 'Error'


class ServiceStatus(BaseModel):
    inference: StatusEnum
    preprocess: StatusEnum
    postprocess: StatusEnum
    service: StatusEnum = StatusEnum.ok


def create_app(model, json_schema, resp_schema, config):
    api = SpecTree('falcon', title=config.name, version=config.version)
    app = falcon.API()
    logger = logging.getLogger(__name__)

    class Homepage:
        def on_get(self, req, resp):
            resp.media = {
                'health check': {'/health': 'GET'},
                'inference': {'/inference': 'POST'},
                'API document': {'/apidoc/redoc': 'GET', '/apidoc/swagger': 'GET'}
            }

    class HealthCheck:
        def __init__(self, model):
            self.model = model

        @api.validate(resp=Response(HTTP_200=ServiceStatus))
        def on_get(self, req, resp):
            """
            Health check
            """
            status = ServiceStatus(
                inference=StatusEnum.ok,
                preprocess=StatusEnum.ok,
                postprocess=StatusEnum.ok,
            )
            logger.debug(str(status))
            resp.media = status.dict()

    class Inference:
        def __init__(self, model):
            self.model = model

        @api.validate(json=json_schema, resp=Response(HTTP_200=resp_schema))
        def on_post(self, req, resp):
            """
            Deep learning model inference
            """
            data = self.model.preprocess(req.context.json)
            data = self.model.inference(data)
            data = self.model.postprocess(data)
            logger.debug(str(data))
            resp.media = data

    app.add_route('/', Homepage())
    app.add_route('/health', HealthCheck(model))
    app.add_route('/inference', Inference(model))
    api.register(app)
    return app
