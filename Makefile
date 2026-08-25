.PHONY: install test lint synthetic privacy

install:
	python -m pip install -e '.[dev,benchmark]'

test:
	pytest

lint:
	ruff check .

synthetic:
	ataad-pop make-synthetic --output data/synthetic_ataad.csv --n 360

privacy:
	ataad-pop privacy-check --root .

