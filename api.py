import uuid
from typing import List
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from utils.inference import infer_model, crop_face_with_expand
import torch
import os
from models.make_network import get_network, load_checkpoint
from configs.cfg import _C as cfg
import sys, traceback
from utils.logging import get_logger
from utils.utils import load_form_data, load_from_local, download_file_from_urls3
from PIL import Image
import tempfile
import datetime, time
from ultralytics import YOLO


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


def infer_api(net, cfg, device, file_name, img, yolo_face_model=None, net_face_crop=None, threshold=0.5, YOLO_Det=False, conf_det=0.5):
    pil_image = Image.fromarray(img)
    prob1 = infer_model(net, cfg, device, img=pil_image)
        
    if prob1[0] > threshold:
        if YOLO_Det:
            img_crop, _ = crop_face_with_expand(img=pil_image, yolo_face_model=yolo_face_model, device=device, conf=conf_det)
            prob3 = infer_model(net_face_crop, cfg, device, img=img_crop)
            prob = prob3[0]
    else:
        prob = prob1[0]
    
    return {
        'source': file_name,
        'prob': "{:.4f}".format((1 - prob) * 100),
        'label': 'live' if prob > threshold else 'spoof',
        'is_spoof': False if prob > threshold else True,
    }


# --- Device ---
device = torch.device(f"cuda:{os.getenv('DEVICE', default='0')}" if torch.cuda.is_available() else "cpu")
logger.info(f"device : {device}")

# --- setup -----
logger.info("load checkpoint is {}".format(os.getenv("WEIGHT", default="best.pt")))
net1 = get_network(cfg=cfg, device=device, backbone=os.getenv("BACKBONE", default="ViT-B/16"))
net1 = load_checkpoint(net1, weight_path=os.getenv("WEIGHT", default="best.pt"))
net1.to(device)
logger.info("load checkpoint is done! {}".format(os.getenv("WEIGHT", default="best.pt")))

logger.info("load checkpoint is {}".format(os.getenv("WEIGHT_FACE", default="best.pt")))
net2 = get_network(cfg=cfg, device=device, backbone=os.getenv("BACKBONE", default="ViT-B/16"))
net2 = load_checkpoint(net1, weight_path=os.getenv("WEIGHT_FACE", default="best.pt"))
net2.to(device)
logger.info("load checkpoint is done! {}".format(os.getenv("WEIGHT_FACE", default="best.pt")))

logger.info("load checkpoint is {}".format(os.getenv("WEIGHT_YOLO_DET", default="best.pt")))
yolo_face_model = YOLO(os.getenv("WEIGHT_YOLO_DET", default="best.pt"))
logger.info("load checkpoint is done! {}".format(os.getenv("WEIGHT_YOLO_DET", default="best.pt")))

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
async def api_vft_ekyb(
        request: Request, 
        files: List[UploadFile] = File(None), 
        urls: list[str] = Form(None), 
        threshold: float = Form(0.5),
        threshold_det: float = Form(0.5),
        yolo_det: bool = Form(True)
    ):
    
    uid, results = uuid.uuid1(), []
    data = {
        "uuid": str(uid),
        "status_code": 200,
        "detail": "success",
        "spoof": False,
        "spoofs": [],
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latency_ms": 0.0,
        
    }
    partner = request.headers.get("X-partner", "VFT-Unknown")
    source = request.headers.get("source", "VFT-Unknown")
    xgw_id = request.headers.get("Xgw-Request-Id", str(uid))
    
    logger.info(
        "ID: {} input data: files={}, url={}, partner={}, source={}, xgw_id={}, threshold: {}, threshold_det: {}, yolo_det: {}".format(
            uid, files, urls, partner, source, xgw_id, threshold, threshold_det, yolo_det
        )
    )

    st_time = time.time()
    invalid_files, invalid_urls = check_input(uid, files, urls, data)
    
    if not invalid_files: # from file
        images, files_name = await load_form_data(files=files, logger=logger, uid=uid)
        for img, file_name in zip(images, files_name):
            output = infer_api(net=net1, cfg=cfg, device=device, file_name=file_name, img=img, YOLO_Det=yolo_det, 
                               threshold=threshold, yolo_face_model=yolo_face_model, net_face_crop=net2)
            if output['is_spoof']:
                data['spoof'] = True
            results.append(output)

    elif not invalid_urls: # from url
        path_files = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            for url in urls:
                file_path, _ = await download_file_from_urls3(url=url, save_path=tmpdirname, uid=uid, timeout=15)
                path_files.append(file_path)
            images, files_name = await load_from_local(files_path=path_files, logger=logger, uid=uid)
            for img, file_name in zip(images, files_name):
                output = infer_api(net=net1, cfg=cfg, device=device, file_name=file_name, img=img, YOLO_Det=yolo_det,
                                   threshold=threshold, yolo_face_model=yolo_face_model, net_face_crop=net2)
                if output['is_spoof']:
                    data['spoof'] = True
                results.append(output)
            
    else:
        data['detail'] = 'HTTP 400 BAD REQUEST'
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=data)
    ed_time = time.time()
    data['spoofs'] = results
    data['latency_ms'] = ed_time - st_time
    return JSONResponse(content=data, status_code=status.HTTP_200_OK)
