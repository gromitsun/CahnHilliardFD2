CC = clang++
CFLAGS = -Wall -O3 -std=c++11 -I../../cl_common/
LFLAGS = -framework OpenCL -L../../cl_common/ -lclcommon
EXE = main.out
SRC_FILES = $(wildcard *.cpp)
OBJ_FILES = $(SRC_FILES:.cpp=.o)


all: $(EXE)
	export DYLD_LIBRARY_PATH=$$DYLD_LIBRARY_PATH:/Users/yue/Dropbox/Documents/Work/OpenCL/cl_common/
$(EXE): $(OBJ_FILES)
	$(CC) -o $(EXE) $(OBJ_FILES) $(LFLAGS)
$(OBJ_FILES): $(SRC_FILES)
	$(CC) -c $(SRC_FILES) $(CFLAGS)
clean:
	rm -fv $(EXE)
	rm -fv *.o
	rm -fv *.gch
