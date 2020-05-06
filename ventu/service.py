from enum import Enum
import logging

import falcon
from falcon import media
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


def create_app(infer, req_schema, resp_schema, use_msgpack, config):
    if use_msgpack:
        handlers = media.Handlers({
            'application/msgpack': media.MessagePackHandler(),
        })
        app = falcon.API(media_type='application/msgpack')
        app.req_options.media_handlers = handlers
        app.resp_options.media_handlers = handlers
    else:
        app = falcon.API()

    api = SpecTree('falcon', title=config.name, version=config.version)
    logger = logging.getLogger(__name__)

    class Homepage:
        def on_get(self, req, resp):
            logger.debug('return service endpoints')
            resp.media = {
                'health check': {'/health': 'GET'},
                'inference': {'/inference': 'POST'},
                'API document': {'/apidoc/redoc': 'GET', '/apidoc/swagger': 'GET'}
            }

    class HealthCheck:
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
        @api.validate(json=req_schema, resp=Response(HTTP_200=resp_schema))
        def on_post(self, req, resp):
            """
            Deep learning model inference
            """
            data = req.context.json
            logger.debug(str(data))
            resp.media = infer(data)

    app.add_route('/', Homepage())
    app.add_route('/health', HealthCheck())
    app.add_route('/inference', Inference())
    api.register(app)
    return app
