CXX = /usr/local/Cellar/llvm/3.9.0/bin/clang++
CXXFLAGS = -O3 -fopenmp -march=native --std=c++14
LDFLAGS += -L/usr/local/Cellar/llvm/3.9.0/lib/ -fopenmp

demo: SVD.o SVDImpl.o movielens_demo.o
	$(CXX) $(LDFLAGS) $^ -o $@

.PHONY: clean
clean:
	$(RM) -rf *.o *.dSYM demo
