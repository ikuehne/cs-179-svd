CXXFLAGS = -O3 --std=c++14

demo: SVD.o SVDImpl.o movielens_demo.o
	$(CXX) $(LDFLAGS) $^ -o $@

.PHONY: clean
clean:
	$(RM) -rf *.o *.dSYM demo
