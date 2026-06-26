CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -O2
LDFLAGS ?= -ldl

BIN = repro_ssyrk
SRC = repro_ssyrk.c
MATRIX = testdata/A.bin

.PHONY: all clean data run

all: data $(BIN)

data: generate_matrix.py
	python3 generate_matrix.py

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: all
	./$(BIN)

clean:
	rm -f $(BIN)
