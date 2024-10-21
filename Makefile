.PHONY: help
help:
	@echo 'make <target>'
	@echo ''
	@echo 'Targets:'
	@echo '    help          Show this help'
	@echo '    pre-commit    Run pre-commit checks'
	@echo ''
	@echo '    configure     Configure the environment'
	@echo '    install       Install required Python packages'
	@echo ''

.PHONY: pre-commit
pre-commit:
	@pre-commit run -a

.PHONY: configure
configure:
	@export PYTHONPATH="./src"

.PHONY: install
install:
	@sudo apt-get install python3-pip
	@sudo pip3 install -r requirements.txt --break-system-packages
