.PHONY: install test lint clean

install:
	pip install -e ".[dev,train,eval]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check leander_tts/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
