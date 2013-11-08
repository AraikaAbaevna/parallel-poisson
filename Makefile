#
# Makefile
#
# Compiles the poisson program 
#

# Configuration
CC=mpicc
LDFLAGS=-O0 -lm
DEBUG=-g -Wall -pedantic -std=c99

all: poisson

poisson: poisson.c
	$(CC) $^ -o $@ $(LDFLAGS)

clean: 
	rm -rf *.o poisson

clean-tmp:
	rm -rf *.dat *.dSYM/ 
