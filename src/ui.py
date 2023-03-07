import logging
import time

import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests


st.set_page_config(
    page_title="Vehicle License Plate Recognition",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('停车场车牌识别')

url = 'http://localhost:8000'
license_plate_recog = '/vehicle/service/vehicle_license_plate_recog'
license_plate_det = '/vehicle/service/vehicle_license_plate_det'

# @st.cache(ttl=60*5, max_entries=10)
def process(image, server_url:str):
    '''
    :param image:
    :param server_url:
    :return:
    '''
    m = MultipartEncoder(fields={'file':('filename', image, 'image/jpeg/jpg/png/bmp')})
    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type':m.content_type},
                      verify=False)
    return r


def vehicle_license_plate_recog():
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            image = st.file_uploader('上传车辆正面有车牌图片', type=['png', 'jpg', 'jpeg', 'bmp'])
            text_json = None
            det_img = None
            if image is None:
                st.write('上传车辆正面有车牌图片！')
            else:
                st.image(image, caption='原图')
                if st.button('✨检测识别'):
                    start_time = time.time()
                    text_json = process(image, url + license_plate_recog)
                    det_img = process(image, url + license_plate_det)
                    end_time = time.time()
                    logging.info('Vehicle License Plate Recognition: {}'.format(end_time - start_time))
        with col2:
            if (text_json is not None) and (det_img is not None):
                start_time = time.time()
                result_json = text_json.json()
                text = result_json['text']
                with st.container():
                    st.markdown('<table><tr><td bgcolor=DarkSeaGreen>车牌检测</td></tr></table>', unsafe_allow_html=True)
                    # 展示原图片
                    st.image(det_img.content)
                    # st.markdown(display_image, unsafe_allow_html=True)
                    if (len(text) != 0):
                        st.markdown('<table><tr><td bgcolor=DarkSeaGreen>车牌识别</td></tr></table>', unsafe_allow_html=True)
                        st.text(text)
                end_time = time.time()
                logging.info('ui time: {}'.format(end_time - start_time))

option_task = st.sidebar.radio('请选择要执行的任务', ('停车场车牌识别',))
if option_task == '停车场车牌识别':
    vehicle_license_plate_recog()

