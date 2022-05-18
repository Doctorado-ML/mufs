SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage deps help lint push test build

coverage:  ## Run tests with coverage
	coverage erase
	coverage run -m unittest -v mufs.tests
	coverage report -m

deps:  ## Install dependencies
	pip install -r requirements.txt

lint:  ## Lint and static-check
	black mufs
	flake8 mufs
	mypy mufs

push:  ## Push code with tags
	git push && git push --tags

test:  ## Run tests
	python -m unittest -v mufs.tests

build:  ## Build package
	rm -fr dist/*
	rm -fr build/*
	python setup.py sdist bdist_wheel

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
