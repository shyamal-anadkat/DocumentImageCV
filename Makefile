install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black scripts/*.py data/*.py

lint:
	pylint --disable=R,C scripts/*.py data/*.py

clean: format lint
all: install format lint
