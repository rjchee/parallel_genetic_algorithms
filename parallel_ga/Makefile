APP_NAME=ga

CXX=g++
CXXFLAGS=-m64 -std=c++11 -I. -I../common -O3 -Wall

COMMONDIR=../common
OBJS=ga.o

default: $(APP_NAME)


# Compile for CPU
cpu: CXX = g++ -m64 -std=c++11
cpu: CXXFLAGS = -I. -I../common -O3 -Wall -fopenmp -Wno-unknown-pragmas

# Compilation Rules
$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

cpu: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(APP_NAME) $(OBJS)

%.o: %.cpp $(COMMONDIR)/CycleTimer.h
	$(CXX) $< $(CXXFLAGS) -c -o $@


clean:
	/bin/rm -rf *~ *.o $(APP_NAME)

