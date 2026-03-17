FROM vft-fas-base:latest

WORKDIR /app/sources
COPY . .
# RUN pip install -r requirements.txt
ENV TZ=Asia/Ho_Chi_Minh
