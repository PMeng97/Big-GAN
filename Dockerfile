FROM anibali/pytorch:cuda-10.0

WORKDIR /ssd-det

COPY requirement.txt /ssd-det
RUN sudo apt-get update
RUN sudo apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r ./requirement.txt

COPY . /ssd-det
RUN sudo chmod -R a+rwx /ssd-det/

RUN mkdir uploadsD
RUN sudo chmod -R a+rwx uploadsD

CMD ["python" ,"-u","app.py"]~
