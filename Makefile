OCULAR_HEADER = src/ocular.h
GRADIENT_HEADER = src/gradient.h
LINESEARCH_HEADER = src/lineSearch.h
HALF_HEADER = src/halfUtils.h

default: program

half.o:  src/halfUtils.cpp $(HALF_HEADER)
	@echo Compiling halfUtils.cpp
	mpiicc -c src/halfUtils.cpp -o half.o

gradient.o:  src/gradient.cpp $(GRADIENT_HEADER)
	@echo Compiling gradient.cpp
	mpiicc -c src/gradient.cpp -o gradient.o

ocular.o:  src/ocular.cpp $(OCULAR_HEADER)
	@echo Compiling ocular.cpp
	mpiicc -c src/ocular.cpp -o ocular.o

lineSearch.o:  src/lineSearch.cpp $(LINESEARCH_HEADER)
	@echo Compiling lineSearch.cpp
	mpiicc -c src/lineSearch.cpp -o lineSearch.o

main.o: src/main.cpp
	@echo Compiling main.cpp
	mpiicc -c src/main.cpp -o main.o

data.o: src/gen.cpp
	@echo Compiling data generator
	icc -c src/gen.cpp -o data.o

program: half.o gradient.o lineSearch.o ocular.o main.o data.o
	@mkdir -p bin
	@echo Finishing compilation
	icc data.o -o bin/data
	mpiicc -qopenmp half.o ocular.o lineSearch.o gradient.o main.o -o bin/ocular
	@echo Removing object files
	@rm -f *.o

clean:
	@echo Cleaning up
	@rm -f *.o
	@rm -rf bin

