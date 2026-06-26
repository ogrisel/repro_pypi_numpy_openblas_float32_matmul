CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -O2
LDFLAGS ?= -ldl

BIN = repro_ssyrk
SRC = repro_ssyrk.c
MATRIX = testdata/A.bin

NUMPY_WHEEL_URL = https://files.pythonhosted.org/packages/85/4b/953118a730ee3b35e28645e0eb4cf9beec5bdbb954e1ac2f5fcefba6bbc3/numpy-2.5.0-cp313-cp313-macosx_11_0_arm64.whl
VENV = .venv
VENV_PYTHON = $(VENV)/bin/python
VENV_PIP = $(VENV)/bin/pip

.PHONY: all clean data run venv

all: data $(BIN)

data: generate_matrix.py
	python3 generate_matrix.py

venv: $(VENV)/bin/python
	$(VENV_PIP) install $(NUMPY_WHEEL_URL)
	@DYLIB=$$($(VENV_PYTHON) -c 'import numpy, pathlib; print(next(pathlib.Path(numpy.__file__).parent.joinpath(".dylibs").glob("libscipy_openblas64_*.dylib")))'); \
	echo "Installed OpenBLAS dylib: $$DYLIB"; \
	echo ""; \
	echo "Run: source .venv/bin/activate"; \
	echo "Then: export OPENBLAS_DYLIB=\"$$DYLIB\""

$(VENV)/bin/python:
	python3.13 -m venv $(VENV)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: all
	./$(BIN)

clean:
	rm -f $(BIN)
