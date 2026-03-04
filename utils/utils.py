import httpx, aiofiles
import asyncio
import datetime
import os
import pathlib
import sys
import tempfile
import traceback
from urllib.parse import urlparse
from utils.logging import get_logger
from utils.exp import DownloadError
import cv2
import numpy as np
import pyheif
from PIL import Image


logger = get_logger()


def lower_img_modes(img_modes):
    """
    Lowercase danh sách định dạng ảnh
    """   
    return (x.lower() for x in img_modes)


async def read_from_binary_file_convert_heic_to_numpy(file_data):
    heif_file = pyheif.read_heif(file_data)
    # Dành cho load form data
    # Convert HEIC to PIL image
    img = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
    return img


# chuyển định dạng heic sang dạng jpg
async def read_from_path_file_convert_heic_to_numpy(path_img: str):
    # Dành cho url heic
    heif_file = pyheif.read(path_img)
    img = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
    return img


async def download_file_from_urls3(url, save_path, uid, timeout=5):
    try:
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)
        file_path = os.path.join(save_path, file_name)
 
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
 
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for 4xx, 5xx
 
            async with aiofiles.open(file_path, 'wb') as out_file:
                await out_file.write(response.content)
 
        return file_path, pathlib.Path(file_name).suffix
 
    except httpx.HTTPStatusError as e:
        logger.error(f"ID {uid} >>> ERROR HTTP: {e.response.status_code}, URL: {url}")
        raise DownloadError(f"http_error_{e.response.status_code}", url)
 
    except httpx.TimeoutException:
        logger.error(f"ID {uid} >>> ERROR: Timeout while downloading file: {url}")
        raise DownloadError("timeout", url)
 
    except httpx.RequestError as e:
        logger.error(f"ID {uid} >>> ERROR url: {url}, Message RequestError: {str(e)}")
        raise DownloadError("request_error", url)
 
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        tb_info = traceback.extract_tb(exc_tb)
        logger.error(
            f"ID {uid} >>> ERROR download file: {save_path}, Message Error: {str(e)}, "
            f"exc_type: {exc_type}, exc_obj: {exc_obj}, exc_tb: {exc_tb}, tb_info: {tb_info}"
        )
        raise DownloadError("unknown_error", url)
    
    
async def load_form_data(files, logger, uid):
    images, files_name = [], []
    for file in files:
        try:
            if pathlib.Path(file.filename).suffix.lower() in lower_img_modes(os.getenv("IMG_MODE", default=".jpg,.png,.JPEG,.jpeg,.jfif").split(",")): # ['.jpg', '.png', '.JPEG', '.jpeg', '.jfif']
                logger.info("IMAGE MODE : ID {} >>> read file >>> {}". format(str(uid), file.filename))
                content = file.file.read()
                image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) # use COLOR_CV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                files_name.append(file.filename)
            
            elif pathlib.Path(file.filename).suffix.lower() in lower_img_modes(os.getenv("IPHONE_MODE", default=".heic").split(",")):
                logger.info("IMAGE IPHONE MODE : ID {} >>> read file >>> {}". format(str(uid), file.filename))
                content = file.file.read()
                image = await read_from_binary_file_convert_heic_to_numpy(file_data=content) # use COLOR_CV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                files_name.append(file.filename)
                
            logger.info("ID {} >>> read done >>> {}". format(str(uid), file.filename))
        except Exception as e :
            exc_type, exc_obj, exc_tb = sys.exc_info()
            tb_info = traceback.extract_tb(exc_tb)
            logger.error("ID {} >>> ERROR inference file: {}, Message Error: {}, exc_type: {}, exc_obj: {}, \
                         exc_tb: {}, tb_info: {}". format(str(uid), file, str(e), exc_type, exc_obj, exc_tb, tb_info))
        finally:
            file.file.close()
    return images, files_name


async def load_from_local(files_path, logger, uid):
    images, files_name = [], []
    for file in files_path:
        try:
            if os.path.exists(file):
                if pathlib.Path(os.path.basename(file)).suffix.lower() in lower_img_modes(os.getenv("IMG_MODE", default=".jpg,.png,.JPEG,.jpeg,.jfif").split(",")):
                    logger.info("IMAGE MODE : ID {} >>> read file >>> {}". format(str(uid), file))
                    image = cv2.imread(file, cv2.IMREAD_COLOR) # use COLOR_CV
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    files_name.append(os.path.basename(file))
                 
                elif pathlib.Path(os.path.basename(file)).suffix.lower() in lower_img_modes(os.getenv("IPHONE_MODE", default=".heic").split(",")):
                    logger.info("IMAGE IPHONE MODE : ID {} >>> read file >>> {}". format(str(uid), file))
                    image = await read_from_path_file_convert_heic_to_numpy(path_img=file) # use COLOR_CV
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    files_name.append(os.path.basename(file))
                    pass
                logger.info("ID {} >>> read done >>> {}". format(str(uid), file))
            else:
                logger.error("ID: {} >>> Can't find file: {}". format(str(uid), file))
                pass
        except Exception as e :
            exc_type, exc_obj, exc_tb = sys.exc_info()
            tb_info = traceback.extract_tb(exc_tb)
            logger.error("ID {} >>> ERROR inference file: {}, Message Error: {}, exc_type: {}, exc_obj: {}, \
                        exc_tb: {}, tb_info: {}". format(str(uid), file, str(e), exc_type, exc_obj, exc_tb, tb_info))
    return images, files_name