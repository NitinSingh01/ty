FROM python:3.7-alpine3.9
# Based on https://github.com/jazzdd86/alpine-flask

# Install base components
RUN apk add gcc g++ musl-dev libffi-dev openssl-dev

# Dir setup
ENV APP_DIR /app
RUN mkdir ${APP_DIR}
VOLUME ${APP_DIR}
WORKDIR ${APP_DIR}

# Install requirements early so we can change code and re-build quickly
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install flask && \
    pip3 install -r requirements.txt

# expose http port
EXPOSE 8080

# copy config files into filesystem
COPY src .

# exectute start up script
ENTRYPOINT ["python3", "Project.py"]
