import uvicorn
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import io
import logging
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from src.model_path import plate_detect_path
from src.model_path import ocr_model_det_pp_path, ocr_model_cls_pp_path, ocr_model_rec_pp_path, \
    ocr_model_char_pp_path
from src.vehicle_license_plate_detection import cut_license, detect_pre_precessing, post_precessing, draw_det_img
from src.vehicle_license_plate_ocr import parse_ocr_opt, load_ocr_model
from PIL import Image
import cv2
import numpy as np
import onnxruntime


# 1.license plate detect
providers = ['CPUExecutionProvider']
session_detect = onnxruntime.InferenceSession(plate_detect_path, providers=providers)

# 2.license plate ocr
ocr_opt = parse_ocr_opt(ocr_model_det_pp_path, ocr_model_cls_pp_path, ocr_model_rec_pp_path, ocr_model_char_pp_path)
ocr_model = load_ocr_model(ocr_opt)


app = FastAPI(
    title="Vehicle License Plate Recognition",
    description="""Obtain object value out of image
                    and return result""",
    version="v1"
    # docs_url=None,
    # redoc_url=None
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/vehicle/service/vehicle_license_plate_det')
def license_plate_det(file: bytes = File(...)):
    '''
    vehicle license plate recognition
    :param file：
    :return:
    '''
    if (file is None) or (file.strip() == ''):
        raise HTTPException(status_code=410, detail={'code': 410, 'message': '上传数据不能为空！'})
    try:
        img = Image.open(io.BytesIO(file))
    except:
        raise HTTPException(status_code=420,
                            detail={'code': 420, 'message': '无法识别图像文件或上传的图片格式错误，支持的图片格式为：PNG、JPG、JPEG、BMP，请进行转码或更换图片！'})
    width, height = img.size
    logging.info('input image size ： width {}， height {}'.format(width, height))
    img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img_size = (640, 640)
    img_process, r, left, top = detect_pre_precessing(img_bgr, img_size)  # 检测前处理
    y_onnx = session_detect.run([session_detect.get_outputs()[0].name], {session_detect.get_inputs()[0].name: img_process})[
        0]
    outputs = post_precessing(y_onnx, r, left, top)  # 检测后处理
    img_result = draw_det_img(img_bgr, outputs.tolist()[0][:4])
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    bytes_io = io.BytesIO()
    img_base64 = Image.fromarray(img_result)
    img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")

@app.post('/vehicle/service/vehicle_license_plate_recog')
def license_plate_recog(file: bytes = File(...)):
    '''
    vehicle license plate recognition
    :param file：
    :return:
    '''
    if (file is None) or (file.strip() == ''):
        raise HTTPException(status_code=410, detail={'code': 410, 'message': '上传数据不能为空！'})
    try:
        img = Image.open(io.BytesIO(file))
    except:
        raise HTTPException(status_code=420,
                            detail={'code': 420, 'message': '无法识别图像文件或上传的图片格式错误，支持的图片格式为：PNG、JPG、JPEG、BMP，请进行转码或更换图片！'})
    width, height = img.size
    logging.info('input image size ： width {}， height {}'.format(width, height))
    img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img_size = (640, 640)
    img_process, r, left, top = detect_pre_precessing(img_bgr, img_size)  # 检测前处理
    y_onnx = session_detect.run([session_detect.get_outputs()[0].name], {session_detect.get_inputs()[0].name: img_process})[
        0]
    outputs = post_precessing(y_onnx, r, left, top)  # 检测后处理
    img_cut = cut_license(img_bgr, outputs.tolist()[0][:4])
    result = {}
    dt_boxes, rec_res = ocr_model(img_cut)
    print('text: {}'.format(rec_res))
    txt = [text[0] for text in rec_res]
    result['text'] = ''.join(txt)
    result['code'] = 200
    result['message'] = "success"
    json_body = jsonable_encoder(result)
    print('json_body: {}'.format(json_body))
    return JSONResponse(status_code=200,
                        content=json_body)

if __name__ == '__main__':
	 uvicorn.run(app='vehicle_service:app', host="0.0.0.0", port=8000, reload=True)



