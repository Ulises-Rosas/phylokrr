SHELL := /bin/bash

NAME    = `python3 setup.py --name`
VERSION = `python3 setup.py --version`

all: dist_dir upload

dist_dir:
	python3 -m build .

upload:
	twine upload dist/*
	rm -rf dist
	rm -rf build
	rm -rf $(NAME).egg-info
