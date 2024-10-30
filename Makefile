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
	@sudo apt-get install -y python3-pip
	@sudo pip3 install -r requirements.txt --break-system-packages

.PHONY: test
test:
	@export PYTHONPATH=$(shell pwd)/src && pytest -vv --cov ./src
