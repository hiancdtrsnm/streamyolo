FROM lapidarioz/docker-python-opencv3-dlib

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /app

RUN wget https://pjreddie.com/media/files/yolov3.weights /app/yolov3.weights

RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg /app/yolov3.cfg

COPY . app