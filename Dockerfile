FROM python:3.7
MAINTAINER Utkarsh Srivastava <utkarshsrivastava.cse@gmail.com>
COPY ./requirements.txt /requirements.txt
WORKDIR /
RUN pip3 install --upgrade pip
RUN pip3 install scikit-build
RUN pip3 install -r requirements.txt
COPY . /
CMD [ "gunicorn", "--config", "/conf/gunicorn.conf.py", "app.controller.flask_controller:app" ]
