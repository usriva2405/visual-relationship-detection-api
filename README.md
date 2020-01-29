![Logo of the project](app/static/img/verification-banner.jpg)

# Relationship Visualization
> Identify a visual relationship in a given image

## Installing / Getting started

this is a python project, and should run fine on version >= 3. 
1. Install python 3.x
2. Create a virtual environment for python

    ```shell
    pip3 install virtualenv
    mkdir ~/.virtualenvs
    
    pip3 install virtualenvwrapper
    export WORKON_HOME=~/Envs
    source /usr/local/bin/virtualenvwrapper.sh
    
    workon
    mkvirtualenv audit_logger
    ```
    This setups up a new virtualenv called audit_logger. <br />

3. Install the required libraries for this project

    ```shell
    pip3 install -r requirements.txt
    ```
4. Install MongoDB and configure it in `conf/config.yaml`

### Initial Configuration

Setup mongoDB correct URL in config.yaml/ or provide environment variables in .env for the url

## Developing

In order to work on this further, use following - 

```shell
git clone git@gl.qpayi.com:qpay/audit-logger.git
cd audit-logger/
```

### *Running Code Directly (Non Docker)*

There are 3 ways to run this directly (locally)
1. Use python to run controller directly
    
    ```shell
    python audit/controller/flask_controller.py
    curl http://127.0.0.1:5000      # prints Welcome to Audit-Logger
    ```
    
    If the project has been setup, this prints ***Welcome to Audit-Logger*** on console

2. Using WSGI Server for running app (without config)

    You can also use following for running the app : 
    ```shell
    gunicorn -b localhost:8880 -w 1 app.controller.flask_controller:app
    curl http://127.0.0.1:5000      # prints Welcome to Audit-Logger
    ```
    App would be accessible on http://127.0.0.1:8880<br /><br />

3. Using WSGI Server for running app (with config)

    Use following for running the app : 
    ```shell
    gunicorn -c conf/gunicorn.conf.py --log-level=debug app.controller.flask_controller:app
    gunicorn -c conf/heroku-gunicorn.conf.py --log-level=debug app.controller.flask_controller:app
    curl http://127.0.0.1:5001      # prints Welcome to Audit-Logger
    ```
    App would be accessible on http://0.0.0.0:5001<br /><br />

## Deploying / Publishing

### Docker
For building the project run
```shell
docker build --no-cache -t visual-relationship:latest .
```

For deploying the project run
```shell
DEV
docker run -d -p 5002:5002 --name visual-relationship -e ENVIRONMENT_VAR=DEV visual-relationship:latest
STAGING
docker run -d -p 5003:5003 --name visual-relationship -e ENVIRONMENT_VAR=STAGING visual-relationship:latest
```

hit localhost:5001 on browser to access the project

## Configuration

Must have mongoDB running and accessible on the URL given in config.yaml

## Sample Request-Response
### Image POST as form-data

We can pass images as form-data (local folder uploads) for verification <br />

URL `localhost:5000/verifyface/` <br />
TYPE `POST (form_data)` <br />
HEADER `Content-Type : multipart/form-data` <br />
SAMPLE request key-value pairs
```
base_image : <path>/b1.png
target_image : <path>/b2.png
```
<br/>

### Image POST as json

We can also pass images as URLs (s3-bucket URLs) for verification <br />

URL `localhost:5000//detectobjects` <br />
TYPE `POST` <br />
HEADER `Content-Type : "application/form-data"` <br />
SAMPLE request json
```
{
    base_image : <<image file>>
}
```

### Heroku deployment

For deploying the project run
```shell
heroku container:login
heroku create visualrelation-api
heroku container:push web --app visualrelation-api
heroku open --app visualrelation-api
```

## Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.

## Links

- Repository: git@bitbucket.org:krazykagglers/trainedmodelapi.git
- Issue tracker: Use JIRA
  - In case of sensitive bugs like security vulnerabilities, please contact
    utkarshsrivastava.cse@gmail.com directly instead of using issue tracker.