import uuid
from typing import List
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from infer import infer_model
import torch
import os
from models.make_network import get_network, load_checkpoint
from configs.cfg import _C as cfg
import sys, traceback
from utils.logging import get_logger
from utils.utils import load_form_data, load_from_local, download_file_from_urls3
from PIL import Image
import tempfile


logger = get_logger()


def check_input(uid, files, urls, data):
    try:
        invalid_files, invalid_urls = True, True
        if files is not None and files[0].size > 0:
                invalid_files = False
        elif urls is not None and urls[0].strip() != "":
            invalid_urls = False
        else:
            data["message"] = "Không tìm thấy files hoặc url"
            data["status"] = status.HTTP_400_BAD_REQUEST
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=data)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        tb_info = traceback.extract_tb(exc_tb)
        logger.error("ID {} >>> ERROR inference: Message Error: {}, exc_type: {}, exc_obj: {}, exc_tb: {}, tb_info: {}".
                    format(uid, str(e), exc_type, exc_obj, exc_tb, tb_info))
    return invalid_files, invalid_urls


# --- Device ---
device = torch.device(f"cuda:{os.getenv('DEVICE', default='0')}" if torch.cuda.is_available() else "cpu")
logger.info(f"device : {device}")

# --- setup -----
net = get_network(cfg=cfg, device=device, backbone=os.getenv("BACKBONE", default="ViT-B/16"))
net = load_checkpoint(net, weight_path=os.getenv("WEIGHT", default="best.pt"))
net.to(device)
logger.info("load checkpoint is done!")

# default sync
router = APIRouter()

@router.get("/")
def ping():
    return{
        "VFT-FAS": "Dịch vụ kiểm tra giả mạo gương mặt",
        "message": "Hi there :P",
        "contributor": "Đỗ Mạnh Quang et al.",
    }


@router.post("/vft-fas") # Giấy tờ ĐKKD
async def api_vft_ekyb(request: Request, files: List[UploadFile] = File(None), urls: list[str] = Form(None)):
    
    uid, results = uuid.uuid1(), {}
    data = {
        "uuid": str(uid),
        "results": None,
        "status": 200,
        "message": "thành công",
    }
    partner = request.headers.get("X-partner", "VFT-Unknown")
    source = request.headers.get("source", "VFT-Unknown")
    xgw_id = request.headers.get("Xgw-Request-Id", str(uid))
    
    logger.info(
        "ID: {} input data: files={}, url={}, partner={}, source={}, xgw_id={}".format(
            uid, files, urls, partner, source, xgw_id
        )
    )

    invalid_files, invalid_urls = check_input(uid, files, urls, data)
    
    if not invalid_files: # from file
        images, files_name = await load_form_data(files=files, logger=logger, uid=uid)
        for img, file_name in zip(images, files_name):
            pil_image = Image.fromarray(img)
            prob = infer_model(net, cfg, device, img=pil_image)
            results[file_name] = {
                'prob': prob[0],
                'label': 'live' if prob[0] > 0.5 else 'spoof',
                'is_spoof': False if prob[0] > 0.5 else True,
            }

    elif not invalid_urls: # from url
        path_files = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            for url in urls:
                file_path, _ = await download_file_from_urls3(url=url, save_path=tmpdirname, uid=uid, timeout=15)
                path_files.append(file_path)
            images, files_name = await load_from_local(files_path=path_files, logger=logger, uid=uid)
            for img, file_name in zip(images, files_name):
                pil_image = Image.fromarray(img)
                prob = infer_model(net, cfg, device, img=pil_image)
                results[file_name] = {
                    'prob': prob[0],
                    'label': 'live' if prob[0] > 0.5 else 'spoof',
                    'is_spoof': False if prob[0] > 0.5 else True,
                }
            
    else:
        data['message'] = 'HTTP 400 BAD REQUEST'
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=data)
    data['results'] = results
    return JSONResponse(content=data, status_code=status.HTTP_200_OK)
