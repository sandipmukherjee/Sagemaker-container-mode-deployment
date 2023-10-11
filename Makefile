lint:
	black --check -t py38 -l 120 .
	pylint src/ tests/ --rcfile=pylint.cfg
	flake8 src/ tests/

test:
	PYTHONPATH=%PYTHONPAH:src coverage run --rcfile=setup.cfg -m pytest
	coverage report --rcfile=setup.cfg

mypy:
	mypy --config setup.cfg ./src

clean:
	rm -rf .mypy_cache .pytest_cache .coverage dist

all: lint	test	mypy