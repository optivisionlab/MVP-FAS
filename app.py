import os
from fastapi import APIRouter, FastAPI
from fastapi import Security, Depends
from fastapi.security.api_key import APIKeyHeader
import api as api_fas
import uvicorn
from utils.logging import get_logger
from fastapi import status, HTTPException


logger = get_logger()

app = FastAPI(title='VFT-FAS',
              description="Face Anti Spoof",
              debug=True,
              )


api_key_header = APIKeyHeader(name="token", auto_error=False)


def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header in os.getenv("token", default="sherlock").split(","):
        return api_key_header
    else:
        logger.info("invalid api key : {}".format(api_key_header))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={
            "status": status.HTTP_401_UNAUTHORIZED,
            "message": f"invalid api key : {api_key_header}",
        }) 


api_router = APIRouter()
api_router.include_router(api_fas.router, dependencies=[Depends(get_api_key)], tags=["API_FAS"])
app.include_router(api_router)


if __name__ == "__main__":
    API_HOST = os.getenv("HOST", default="0.0.0.0")
    API_PORT = int(os.getenv('PORT', default="2026"))
    logger.info("HOST : {}, PORT: {}".format(API_HOST, API_PORT))
    uvicorn.run('app:app', host=API_HOST, port=API_PORT)
    pass
