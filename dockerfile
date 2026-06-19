FROM dockerhub.ovp.vn/vft/vft-fas:vb4.0

WORKDIR /app/sources
COPY . .
# RUN pip install -r requirements.txt
# RUN pip install ultralytics==8.4.37
ENV TZ=Asia/Ho_Chi_Minh
