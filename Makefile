.PHONY: install data index golden eval demo clean test

install:
	pip install -e ".[dev]" --break-system-packages || pip install -r requirements.txt --break-system-packages

data:
	python scripts/run_pipeline.py --build-index

golden:
	python scripts/build_golden_eval.py

eval:
	python scripts/run_pipeline.py --eval-only

demo:
	python scripts/demo.py

batch-demo:
	python scripts/demo.py batch data/golden_eval/examples.json --n 20

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
