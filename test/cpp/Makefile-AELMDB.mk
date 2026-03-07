# Simple, clean Makefile for building the three test programs
# Usage:
#   make            # build all
#   make slice      # build rbsr_aelmdb_slice_tests
#   make advanced   # build rbsr_aelmdb_advanced_test
#   make clean

CXX      := g++
# CXXFLAGS := -std=c++20 -g -pg
CXXFLAGS := -std=c++20 -O3 -DNDEBUG

# Include / library paths (as in your commands)
INCLUDES := -I../../cpp \
            -I../../../ \
            -I../../../lmdbxx-aelmdb/include/ \
			-I../../../aelmdb/ \
			-I. -I/opt/homebrew/include/
LDFLAGS  := -L/opt/homebrew/lib/ #-g -pg
LDLIBS   := -lcrypto

# Prebuilt objects you link against
OBJS := ../../../aelmdb/obj/mdb.o ../../../aelmdb/obj/midl.o

# Programs
PROGS := rbsr_aelmdb_slice_tests rbsr_aelmdb_advanced_test

.PHONY: all clean slice advanced
all: $(PROGS)

rbsr_aelmdb_slice_tests: rbsr_aelmdb_slice_tests.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LDLIBS) -o $@

# rbsr_aelmdb_dual_test: rbsr_aelmdb_dual_test.cpp $(OBJS)
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LDLIBS) -o $@

rbsr_aelmdb_advanced_test: rbsr_aelmdb_advanced_test.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LDLIBS) -o $@

# Convenience aliases
slice: rbsr_aelmdb_slice_tests
advanced: rbsr_aelmdb_advanced_test

clean:
	$(RM) $(PROGS)