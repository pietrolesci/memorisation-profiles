sources = src/ scripts/

format:
	ruff format $(sources)

lint:
	ruff check $(sources) --fix
	torchfix $(sources) --fix