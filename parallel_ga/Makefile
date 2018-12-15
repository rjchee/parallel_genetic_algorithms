
EXECUTABLE := ga

CU_FILES   := ga.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
COMMON_DIR=../common
CXX=g++ -m64 -std=c++11
CXXFLAGS=-O3 -Wall -I. -I$(COMMON_DIR)
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
endif
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35 -I. -I$(COMMON_DIR)


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/ga.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp $(COMMON_DIR)/CycleTimer.h
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu $(COMMON_DIR)/CycleTimer.h
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
