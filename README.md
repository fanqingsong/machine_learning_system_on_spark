# Machine Learning System on Spark

## introduction

This project is a full stack Django/React/Redux app that uses token based authentication with Knox.

Then I add Machine Learning features for demostrate the full workflow of the data mining, including the four stage corresponding to four pages:
1. data management
2. data explore
3. model train
4. prediction

The data set is the classic iris data, which is only for demo, and this project is from my interest. so you can reference, but the quality is not assured.

## features

* authentication functions
> login from login page
> register your account
> logout from inner page

* data management
> input iris items
> edit iris items
> delete iris items

* data explore
> inspect attribute distribution through histogram
> inspect sepal distribution through scatter graph
> inspect petal distribution through scatter graph

* model train
> input cluster number
> train a cluster model using spark.ml library
> inspect cluster result through sepal and petal scatter

* prediction
> input iris sepal and petal attributes
> predict iris cluster

## Architecture

Generally, train process is time consumming, and predict process is quick.
So set train flow as async mode, and predict flow as sync mode.

### train flow

* user start model train from browser
* django recieve the "start train" message
* django schedule spark.ml celery process to train model and return to user immediately.
* browser query train status to django
* django query train status from spark.ml celery process.
* django feedback the train over status to browser

### predict flow

* user input prediction features on browser, then click submit
* browser send prediction features to django
* django call prediction api with prediction features
* django feedback the prediction result to browser

```
+---------+            +-------------+            +------------+
|         | start train|             |            |            |
|         +------------>             |   start    |            |
|         |            |             +---train---->            |
|         |  query train             |            |            |
|         +--status---->             |            |            |
|         |            |             +----query -->            |
|         <---train ---+             |    train   |            |
|         |   over     |             |    status  |            |
|  browser|            |   django    |            |  spark.ml  |
|         |            |             |            |  on celery |
|         |  predict   |             |            |            |
|         +------------>             |    predict |            |
|         |            |             +----------->+            |
|         <--predict---+             |            |            |
|         |  result    |             |            |            |
|         |            |             |            |            |
|         |            |             |            |            |
+---------+            +-------------+            +------------+

```

## technology stack

category | name | comment
---------|----------|---------
 frontend | reactjs | frontend framework
 frontend | redux | state management
 frontend | react-C3JS | D3 based graph tool
 frontend | react-bootstrap | style component library
 frontend | data-ui | react data visualization tool
 backend | django | backend framework
 backend | django-rest-knox | authentication library
 backend | djangorestframework | restful framework
 backend | spark.ml | machine learning tool

## Install

### python3.5 install

```bash
wget https://www.python.org/ftp/python/3.5.3/Python-3.5.3.tgz
tar -xvzf Python-3.5.3.tgz
cd Python-3.5.3
./configure
make && make install
```

### Install python package
```bash
pip3 install -r requirements.txt
```

### install redis
```bash
sudo apt-get install redis-server
```

### Spark Install
[reference](http://dblab.xmu.edu.cn/blog/1307-2/)


## Run

### start redis server

```bash
redis-server
```

### frontend build

```bash
cd ./frontend
# Install dependencies
npm install
# Build for production
npm run build
```

### start celery worker for django

```bash
cd ./backend
celery worker -A config -l debug
```

### start Server on localhost:8000

```bash
cd ./backend
python3 manage.py runserver
```

## snapshot

### login page
![avatar](./snapshot/loginpage.png)

### data management page
![avatar](./snapshot/dataManagement.png)

### data explore page
![avatar](./snapshot/explore.png)

### model train page
![avatar](./snapshot/train.png)

### prediction page
![avatar](./snapshot/predict.png)


## Memo:

```bash
（1）use this command to make new requirements.txt, if new package is installed.
pip3 freeze > requirements.txt

（2）the following command for ml test
cd ml_celery_research
# start celery worker for test
celery -A celerytask worker -l info

# test celery worker running
python3 celerytask.py

# test kmeans stand-alone code
python3 kmeans_demo.py

# start celery worker for ml
celery -A iris_cluster worker -l info

# test ml celery worker running
python3 iris_cluster.py

(3) python version change

sudo pip3 uninstall pip && sudo apt install python3-pip --reinstall
or
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

from
https://zhuanlan.zhihu.com/p/37473690

(5) python3.5 install
wget https://www.python.org/ftp/python/3.5.3/Python-3.5.3.tgz
tar -xvzf Python-3.5.3.tgz
cd Python-3.5.3
./configure
make && make install

from
https://www.jianshu.com/p/95bd2622c8d5

(6) sqlite3 error solution
# sqlite install
wget https://www.sqlite.org/2017/sqlite-autoconf-3170000.tar.gz --no-check-certificate
tar xf  sqlite-autoconf-3170000.tar.gz
cd sqlite-autoconf-3170000/
./configure --prefix=/usr/local/sqlite3 --disable-static --enable-fts5 --enable-json1 CFLAGS="-g -O2 -DSQLITE_ENABLE_FTS3=1 -DSQLITE_ENABLE_FTS4=1 -DSQLITE_ENABLE_RTREE=1"
make && make install

python install
cd Python-3.5.3
LD_RUN_PATH=/usr/local/sqlite3/lib ./configure LDFLAGS="-L/usr/local/sqlite3/lib" CPPFLAGS="-I /usr/local/sqlite3/include"
LD_RUN_PATH=/usr/local/sqlite3/lib make
LD_RUN_PATH=/usr/local/sqlite3/lib sudo make install

from
https://www.cnblogs.com/i1991/p/9497259.html

(7) python specific version install & pipenv usage
apt-get update
apt-get install python3.7
pip3 install pipenv
pipenv install

(8) django and celery integration
https://www.pythonf.cn/read/7143
https://www.cnblogs.com/wdliu/p/9530219.html

```