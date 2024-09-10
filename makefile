sources = src/ scripts/ multipythias/

format:
	ruff format $(sources)

lint:
	ruff check $(sources) --fix
	torchfix $(sources) --fix