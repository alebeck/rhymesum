LLAMADIR := llama
BLAKE3DIR := blake3

INCLUDES := -I$(LLAMADIR)/include -I$(LLAMADIR)/ggml/include -I$(BLAKE3DIR)
CXXFLAGS := -std=c++11 -O3 -Wall -Wextra -pedantic -pthread $(INCLUDES)

LIB := .
LIBLLAMA := $(LIB)/libllama.so
LIBGGML := $(LIB)/libggml.so
LIBBLAKE3 := $(LIB)/libblake3.so

all: rhymesum

$(LIBLLAMA):
	$(MAKE) -C $(LLAMADIR)
	cp $(LLAMADIR)/*.so $(LIB)

$(LIBGGML): $(LIBLLAMA)

$(LIBBLAKE3):
	$(MAKE) -C $(BLAKE3DIR)
	cp $(BLAKE3DIR)/*.so $(LIB)

rhymesum: main.o words.o $(LIBLLAMA) $(LIBGGML) $(LIBBLAKE3)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.cpp words.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

words.o: words.cpp words.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rvf *.o rhymesum
	rm -vf $(LIB)/*.so
	$(MAKE) -C $(LLAMADIR) clean
	$(MAKE) -C $(BLAKE3DIR) clean
