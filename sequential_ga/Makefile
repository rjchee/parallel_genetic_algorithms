APP_NAME=ga

OBJS=ga.o

default: $(APP_NAME)


# Compile for CPU
cpu: CXX = g++ -m64 -std=c++11
cpu: CXXFLAGS = -I. -O3 -Wall -fopenmp -Wno-unknown-pragmas

# Compilation Rules
$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

cpu: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(APP_NAME) $(OBJS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@


clean:
	/bin/rm -rf *~ *.o $(APP_NAME)

