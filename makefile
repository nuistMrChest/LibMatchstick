BUILD_DIR=./build
OBJ_DIR=./obj

CUDAFLAGS=-O2 -Xcompiler -fPIC

CPP=./src/cpp

CAPI=./src/c_api

CPPINCLUDE=./include/matchstick

CINCLUDE=./include/matchstick_c

OBJS=$(OBJ_DIR)/activation.o $(OBJ_DIR)/layer.o $(OBJ_DIR)/loss.o $(OBJ_DIR)/matrix.o $(OBJ_DIR)/network.o $(OBJ_DIR)/tensor_3d.o $(OBJ_DIR)/tensor_4d.o $(OBJ_DIR)/matrix_capi.o $(OBJ_DIR)/network_capi.o $(OBJ_DIR)/tensor_3d_capi.o $(OBJ_DIR)/tensor_4d_capi.o

all:$(BUILD_DIR)/libmatchstick.so

BUILD:
	mkdir -p $(BUILD_DIR)

OBJ:
	mkdir -p $(OBJ_DIR)

$(BUILD_DIR)/libmatchstick.so:$(OBJS)|BUILD OBJ
	nvcc -shared $(OBJS) -cudart=static -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc -o $(BUILD_DIR)/libmatchstick.so

$(OBJ_DIR)/activation.o:$(CPP)/activation.cu $(CPPINCLUDE)/activation.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CPP)/activation.cu -o $(OBJ_DIR)/activation.o

$(OBJ_DIR)/layer.o:$(CPP)/layer.cu $(CPPINCLUDE)/layer.h $(CPPINCLUDE)/matrix.h $(CPPINCLUDE)/activation.h $(CPPINCLUDE)/tensor_3d.h $(CPPINCLUDE)/tensor_4d.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CPP)/layer.cu -o $(OBJ_DIR)/layer.o

$(OBJ_DIR)/loss.o:$(CPP)/loss.cu $(CPPINCLUDE)/loss.h $(CPPINCLUDE)/matrix.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CPP)/loss.cu -o $(OBJ_DIR)/loss.o

$(OBJ_DIR)/matrix.o:$(CPP)/matrix.cu $(CPPINCLUDE)/matrix.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CPP)/matrix.cu -o $(OBJ_DIR)/matrix.o

$(OBJ_DIR)/network.o:$(CPP)/network.cu $(CPPINCLUDE)/network.h $(CPPINCLUDE)/matrix.h $(CPPINCLUDE)/layer.h $(CPPINCLUDE)/loss.h $(CPPINCLUDE)/tensor_3d.h $(CPPINCLUDE)/tensor_4d.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CPP)/network.cu -o $(OBJ_DIR)/network.o

$(OBJ_DIR)/tensor_3d.o:$(CPP)/tensor_3d.cu $(CPPINCLUDE)/tensor_3d.h $(CPPINCLUDE)/tensor_4d.h $(CPPINCLUDE)/matrix.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CPP)/tensor_3d.cu -o $(OBJ_DIR)/tensor_3d.o

$(OBJ_DIR)/tensor_4d.o:$(CPP)/tensor_4d.cu $(CPPINCLUDE)/tensor_4d.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CPP)/tensor_4d.cu -o $(OBJ_DIR)/tensor_4d.o

$(OBJ_DIR)/matrix_capi.o:$(CAPI)/matrix_capi.cu $(CAPI)/matrix_capi.h $(CINCLUDE)/matrix.h $(CPPINCLUDE)/matrix.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CAPI)/matrix_capi.cu -o $(OBJ_DIR)/matrix_capi.o

$(OBJ_DIR)/network_capi.o:$(CAPI)/network_capi.cu $(CAPI)/network_capi.h $(CAPI)/matrix_capi.h $(CAPI)/tensor_3d_capi.h $(CAPI)/tensor_4d_capi.h $(CINCLUDE)/network.h $(CPPINCLUDE)/activation.h $(CPPINCLUDE)/layer.h $(CPPINCLUDE)/loss.h $(CPPINCLUDE)/network.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CAPI)/network_capi.cu -o $(OBJ_DIR)/network_capi.o

$(OBJ_DIR)/tensor_3d_capi.o:$(CAPI)/tensor_3d_capi.cu $(CAPI)/tensor_3d_capi.h $(CINCLUDE)/tensor_3d.h $(CPPINCLUDE)/tensor_3d.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CAPI)/tensor_3d_capi.cu -o $(OBJ_DIR)/tensor_3d_capi.o

$(OBJ_DIR)/tensor_4d_capi.o:$(CAPI)/tensor_4d_capi.cu $(CAPI)/tensor_4d_capi.h $(CINCLUDE)/tensor_4d.h $(CPPINCLUDE)/tensor_4d.h|OBJ
	nvcc $(CUDAFLAGS) -c $(CAPI)/tensor_4d_capi.cu -o $(OBJ_DIR)/tensor_4d_capi.o

clean:
	rm -rf ./obj
	rm -rf ./build

install:all
	sudo cp ./build/libmatchstick.so /usr/local/lib
	sudo cp -r ./include/* /usr/local/include

uninstall:
	sudo rm /usr/local/lib/libmatchstick.so
	sudo rm -rf /usr/local/include/matchstick
	sudo rm -rf /usr/local/include/matchstick_c








