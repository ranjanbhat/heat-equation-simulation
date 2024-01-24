KOKKOS_PATH = ${HOME}/kokkos
KOKKOS_SRC_PATH = ${KOKKOS_PATH}
SRC = $(wildcard /home/hooded/HeatEquation/heat-equation-simulation/kokkos/*.cpp)
N = 128
vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -O3
LINK = ${CXX}
LDFLAGS = 
EXE = simulation.cuda
KOKKOS_DEVICES = "Cuda,OpenMP"
KOKKOS_ARCH = "SNB,Kepler35"
else
CXX = g++
CXXFLAGS = -O3
LINK = ${CXX}
LDFLAGS =  
EXE = simulation.host
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "SNB"
endif

DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

test: $(EXE)
	./$(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host *.txt

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

# Run the executable if it exists and run with target arguments
run_exe: $(EXE)
	./$(EXE) $(N)
	
run: run_exe
	python3 visualize.py $(N)

test_error: build
	python3 test_accuracy.py