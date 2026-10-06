/* Microbench: random expert-sized reads with controllable host access mode.
 *
 * Measures the patterns Colibri actually cares about: N workers reading random
 * expert-like slabs, buffered or direct/non-cached. The optional access mode
 * distinguishes three host-I/O shapes:
 *
 *   per-thread  one fd/HANDLE per worker (historical iobench default)
 *   shared      all workers pread() the same fd/HANDLE (matches shard tables)
 *   map         persistent read-only file mapping, demand-faulted then memcpy'd
 *
 * usage:
 *   ./iobench <file> [block_MB] [reads] [threads] [direct 0/1]
 *             [per-thread|shared|map]
 *
 * map is intentionally buffered: file mappings participate in the OS page
 * cache. The memcpy makes every byte in the selected range observable, so the
 * benchmark cannot measure a cheap mapping call while deferring the real faults.
 *
 * build: gcc -O2 -fopenmp iobench.c -o iobench
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include "compat.h"
#ifdef _OPENMP
#include <omp.h>
#endif

enum access_mode { ACCESS_PER_THREAD = 0, ACCESS_SHARED = 1, ACCESS_MAP = 2 };

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static const char *access_name(enum access_mode mode) {
    switch (mode) {
        case ACCESS_SHARED: return "shared";
        case ACCESS_MAP: return "map";
        default: return "per-thread";
    }
}

static int parse_access(const char *s, enum access_mode *mode) {
    if (!s || !*s || !strcmp(s, "per-thread") || !strcmp(s, "per_thread")) {
        *mode = ACCESS_PER_THREAD;
        return 0;
    }
    if (!strcmp(s, "shared")) { *mode = ACCESS_SHARED; return 0; }
    if (!strcmp(s, "map") || !strcmp(s, "mmap")) { *mode = ACCESS_MAP; return 0; }
    return -1;
}

/* Windows direct I/O uses a synchronous FILE_FLAG_NO_BUFFERING handle. That
 * is fine when each worker has its own handle, but deliberately exposes the
 * serialization cost in ACCESS_SHARED. */
static int bench_open(const char *path, int *direct) {
#ifdef O_DIRECT
    int fd = open(path, O_RDONLY | (*direct ? O_DIRECT : 0));
    if (fd < 0 && *direct) {
        fprintf(stderr, "O_DIRECT is unavailable (%s); using buffered I/O\n", strerror(errno));
        *direct = 0;
        fd = open(path, O_RDONLY);
    }
#elif defined(_WIN32)
    int fd = *direct ? compat_open_direct(path) : open(path, COMPAT_O_RDONLY);
    if (fd < 0 && *direct) {
        fprintf(stderr, "NO_BUFFERING is unavailable; using buffered I/O\n");
        *direct = 0;
        fd = open(path, COMPAT_O_RDONLY);
    }
#else
    int fd = open(path, O_RDONLY);
#ifdef __APPLE__
    if (*direct && fd >= 0) fcntl(fd, F_NOCACHE, 1);
#else
    if (*direct) {
        fprintf(stderr, "O_DIRECT is unavailable; using buffered I/O\n");
        *direct = 0;
    }
#endif
#endif
    return fd;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s file [blkMB] [n] [threads] [direct 0/1] [per-thread|shared|map]\n", argv[0]);
        return 1;
    }

    long blk = (long)((argc > 2 ? atof(argv[2]) : 19) * 1024 * 1024);
    blk = (blk + 4095) & ~4095L;
    if (blk < 4096) blk = 4096;
    int n = argc > 3 ? atoi(argv[3]) : 64;
    int nth = argc > 4 ? atoi(argv[4]) : 8;
    int direct = argc > 5 ? atoi(argv[5]) : 1;
    enum access_mode mode = ACCESS_PER_THREAD;
    if (argc > 6 && parse_access(argv[6], &mode) != 0) {
        fprintf(stderr, "unknown access mode '%s'\n", argv[6]);
        return 2;
    }
    if (n < 1 || nth < 1) {
        fprintf(stderr, "reads and threads must be positive\n");
        return 2;
    }
    if (mode == ACCESS_MAP && direct) {
        fprintf(stderr, "map mode is page-cache backed; ignoring direct=1\n");
        direct = 0;
    }

    int fd = bench_open(argv[1], &direct);
    if (fd < 0) { perror("open"); return 1; }
#ifdef _WIN32
    off_t sz = compat_fsize(fd);
#else
    off_t sz = lseek(fd, 0, SEEK_END);
#endif
    if (sz < (off_t)blk * 2) {
        fprintf(stderr, "file is too small\n");
        close(fd);
        return 1;
    }

    off_t *offs = malloc((size_t)n * sizeof(*offs));
    if (!offs) { perror("malloc offsets"); close(fd); return 1; }
    srand(1234);
    for (int i = 0; i < n; i++) {
        off_t r30 = ((off_t)rand() << 15) | rand();
        off_t o = (r30 * 4096) % (sz - blk);
        offs[i] = o & ~(off_t)4095;
    }

    compat_ro_map mapped = {0};
    const void *mapped_data = NULL;
    if (mode == ACCESS_MAP) {
        if ((uint64_t)sz > (uint64_t)SIZE_MAX ||
            compat_map_readonly(fd, 0, (size_t)sz, &mapped, &mapped_data) != 0) {
            perror("map");
            free(offs);
            close(fd);
            return 1;
        }
    }

    double t0 = now();
    int64_t tot = 0;
    uint64_t sink = 0;
#pragma omp parallel num_threads(nth) reduction(+:tot,sink)
    {
        void *buf = NULL;
        if (posix_memalign(&buf, 4096, (size_t)blk) != 0 || !buf) {
            fprintf(stderr, "memalign failed\n");
            exit(1);
        }
        int tfd = -1;
        int owns_fd = 0;
        if (mode == ACCESS_PER_THREAD) {
            int d2 = direct;
            tfd = bench_open(argv[1], &d2);
            if (tfd < 0) { perror("open worker"); exit(1); }
            owns_fd = 1;
        } else if (mode == ACCESS_SHARED) {
            tfd = fd;
        }

#pragma omp for schedule(dynamic,1)
        for (int i = 0; i < n; i++) {
            if (mode == ACCESS_MAP) {
                memcpy(buf, (const uint8_t *)mapped_data + offs[i], (size_t)blk);
                tot += blk;
                sink += ((const uint8_t *)buf)[0];
                sink += ((const uint8_t *)buf)[blk - 1];
            } else {
                ssize_t r = pread(tfd, buf, (size_t)blk, offs[i]);
                if (r < 0) {
                    perror("pread");
                } else {
                    tot += r;
                    if (r > 0) {
                        sink += ((const uint8_t *)buf)[0];
                        sink += ((const uint8_t *)buf)[r - 1];
                    }
                }
            }
        }

        if (owns_fd) close(tfd);
        compat_aligned_free(buf);
    }

    double dt = now() - t0;
    printf("%s/%s x%d threads: %d reads x %.4g MB = %.1f GB in %.2fs -> %.2f GB/s (%.1f effective ms/block, check=%llu)\n",
           direct ? "O_DIRECT" : "buffered", access_name(mode), nth, n,
           blk / 1048576.0, tot / 1e9, dt, tot / 1e9 / dt, dt / n * 1000,
           (unsigned long long)sink);

    if (mode == ACCESS_MAP) compat_unmap_readonly(&mapped);
    free(offs);
    close(fd);
    return 0;
}
