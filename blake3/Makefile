UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
UNAME_M := $(shell uname -m)
SRC := blake3.c blake3_dispatch.c blake3_portable.c

ifeq ($(UNAME_P),arm)
	SRC += blake3_neon.c
endif

ifeq ($(UNAME_P),x86)
	SRC += blake3_sse2_x86-64_unix.S blake3_sse41_x86-64_unix.S \
	blake3_avx2_x86-64_unix.S blake3_avx512_x86-64_unix.S
endif

CFLAGS := -shared -O3

all: libblake3.so

libblake3.so: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f libblake3.so
