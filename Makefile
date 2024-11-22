PYTHON = $(shell which python3.10 2>/dev/null)

.PHONY: help
help:
	@echo 'make <target>'
	@echo ''
	@echo 'Targets:'
	@echo '    help       Show this help'
	@echo '    pre-commit Run pre-commit checks'
	@echo ''
	@echo '    install    Install required Python packages'
	@echo '    test       Test Python packages'
	@echo ''

.PHONY: pre-commit
pre-commit:
	@pre-commit run -a

.PHONY: install
install:
ifneq ($(PYTHON),)
	@python3.10 -m ensurepip --upgrade
	@python3.10 -m pip install -r requirements-python3.10.txt
	@python3.10 -m pip install --upgrade setuptools
else
	@echo 'Python 3.10 not found! Some Tensorflow Extended (TFX) features may not work!'
	@sudo python3 -m pip install -r requirements.txt --break-system-packages
	@sudo python3 -m pip install --upgrade setuptools --break-system-packages
endif

.PHONY: test
test:
ifneq ($(PYTHON),)
	@export PYTHONPATH=$(shell pwd)/src && python3.10 -m pytest -vv --cov ./src
else
	@echo 'Python 3.10 not found! Some Tensorflow Extended (TFX) features may not work!'
	@export PYTHONPATH=$(shell pwd)/src && python3 -m pytest -vv --cov ./src
endif
