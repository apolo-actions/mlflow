.PHONY: build

IMAGE_NAME := mlflow-server
IMAGE_TAG := latest

venv:
	python -m venv venv
	. venv/bin/activate; \
	pip install -U pip pip-tools

.PHONY: test-requirements
test-requirements: venv
	. venv/bin/activate; \
	python -m pip install mlflow pytest

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

clean:
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: run-server
run-server:
	docker run --rm -it -p 8080:8080 --name mlflow-test-container -d $(IMAGE_NAME):$(IMAGE_TAG) \
	server --host 0.0.0.0 --port 8080 --artifacts-destination /home --backend-store-uri sqlite:///mydb.sqlite

.PHONY: stop-server
stop-server:
	docker stop mlflow-test-container

.PHONY: test
test: test-requirements
	@rm -rf ./mlruns && \
	. venv/bin/activate && \
	pytest tests

.PHONY: run-tests
run-tests: run-server test stop-server