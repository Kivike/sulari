CPP_FILES = $(wildcard src/*.cpp)  $(wildcard src/*/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
OPENCV = `pkg-config opencv --cflags --libs`
CC_FLAGS := -std=c++11 -g -Wall
LIBS = -L/usr/local/opencv $(OPENCV) -lpthread

INC_DIR = include
INC_PARAMS=$(foreach d, $(INC_DIR), -I$d)

sulari: $(OBJ_FILES)
	echo $(OBJ_FILES)
	g++ $(OPENCV)  $(CC_FLAGS) $(INC_PARAMS) -o $@ $^ $(LIBS)

obj/%.o: src/%.cpp
	g++ $(CC_FLAGS) $(LIBS) $(INC_PARAMS) -c $< -o $@
