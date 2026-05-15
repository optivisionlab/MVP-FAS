FROM vft-fas-base:latest

WORKDIR /app/sources
COPY . .
# RUN pip install -r requirements.txt
RUN pip install ultralytics==8.4.37
ENV TZ=Asia/Ho_Chi_Minh
