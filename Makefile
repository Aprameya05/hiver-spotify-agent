.PHONY: install run eval demo web lint help

help:
    @echo "Usage: make <target>"

install:
    pip install -e . --break-system-packages

run:
    python scripts/run_pipeline.py --max-threads 10000

eval:
    python scripts/run_pipeline.py --eval-only

demo:
    python scripts/demo.py

web:
    uvicorn app_web:app --reload --host 0.0.0.0 --port 8000

lint:
    python -m mypy src/ --ignore-missing-imports --no-error-summary
