OCULAR_HEADER = src/ocular.h
GRADIENT_HEADER = src/gradient.h
LINESEARCH_HEADER = src/lineSearch.h

default: program

gradient.o:  src/gradient.cpp $(GRADIENT_HEADER)
	@echo Compiling gradient.cpp
	mpiCC -c src/gradient.cpp -o gradient.o

ocular.o:  src/ocular.cpp $(OCULAR_HEADER)
	@echo Compiling ocular.cpp
	mpiCC -c src/ocular.cpp -o ocular.o

lineSearch.o:  src/lineSearch.cpp $(LINESEARCH_HEADER)
	@echo Compiling lineSearch.cpp
	mpiCC -c src/lineSearch.cpp -o lineSearch.o

main.o: src/main.cpp
	@echo Compiling main.cpp
	mpiCC -c src/main.cpp -o main.o

data.o: src/gen.cpp
	@echo Compiling data generator
	g++ -c src/gen.cpp -o data.o

program: gradient.o lineSearch.o ocular.o main.o data.o
	@mkdir -p bin
	@echo Finishing compilation
	g++ data.o -o bin/data
	mpiCC -fopenmp ocular.o lineSearch.o gradient.o main.o -o bin/ocular
	@echo Removing object files
	@rm -f *.o

clean:
	@echo Cleaning up
	@rm -f *.o
	@rm -rf bin

