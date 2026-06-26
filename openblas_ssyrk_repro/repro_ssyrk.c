/*
 * Minimal OpenBLAS reproducer: float32 SSYRK on Apple M4 (vortexm4).
 *
 * Matches NumPy 2.5.0's SYRK fast path for C-contiguous float32 A @ A.T:
 *
 *   cblas_ssyrk(RowMajor, Upper, NoTrans, n=300, k=672,
 *               alpha=1, A, lda=672, beta=0, C, ldc=300)
 *
 * On macOS arm64 with OpenBLAS 0.3.33.112.0 vortexm4 dispatch, SSYRK returns
 * wrong results. The equivalent SGEMM call in compare_sgemm() is included as
 * a control and typically passes.
 *
 * Build: see README.md and Makefile.
 */

#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };

#define M 300
#define K 672
#define REFERENCE_MAX 185.7f
#define REFERENCE_RTOL 1e-4f

typedef void (*ssyrk_fn)(int order, int uplo, int trans,
                         long long n, long long k,
                         float alpha, const float *a, long long lda,
                         float beta, float *c, long long ldc);

typedef void (*sgemm_fn)(int order, int trans_a, int trans_b,
                         long long m, long long n, long long k,
                         float alpha, const float *a, long long lda,
                         const float *b, long long ldb,
                         float beta, float *c, long long ldc);

static int read_matrix(const char *path, float *a, size_t n) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "fopen(%s): %s\n", path, strerror(errno));
        return -1;
    }
    size_t got = fread(a, sizeof(float), n, fp);
    fclose(fp);
    if (got != n) {
        fprintf(stderr, "expected %zu floats in %s, got %zu\n", n, path, got);
        return -1;
    }
    return 0;
}

static void mirror_upper_to_lower(float *c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            c[j * n + i] = c[i * n + j];
        }
    }
}

static float matrix_max(const float *c, int n) {
    float mx = -INFINITY;
    int nan_count = 0;
    for (int i = 0; i < n * n; ++i) {
        if (isnan(c[i])) {
            nan_count++;
            continue;
        }
        if (isfinite(c[i]) && c[i] > mx) {
            mx = c[i];
        }
    }
    if (nan_count > 0) {
        printf("  NaN count: %d\n", nan_count);
    }
    return mx;
}

static int matches_reference(float mx) {
    return isfinite(mx) && fabsf(mx - REFERENCE_MAX) <= REFERENCE_RTOL * REFERENCE_MAX;
}

static void reference_gemm_double(const float *a, float *c) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0.0;
            for (int p = 0; p < K; ++p) {
                sum += (double)a[i * K + p] * (double)a[j * K + p];
            }
            c[i * M + j] = (float)sum;
        }
    }
}

static void *load_blas(const char *path, const char *sym, void **handle_out) {
    void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "dlopen(%s): %s\n", path, dlerror());
        return NULL;
    }
    void *fn = dlsym(handle, sym);
    if (!fn) {
        fprintf(stderr, "dlsym(%s): %s\n", sym, dlerror());
        dlclose(handle);
        return NULL;
    }
    *handle_out = handle;
    return fn;
}

static const char *default_blas_path(void) {
    const char *env = getenv("OPENBLAS_DYLIB");
    if (env && env[0]) {
        return env;
    }
    return "libscipy_openblas64_.dylib";
}

int main(int argc, char **argv) {
    const char *matrix_path = "testdata/A.bin";
    const char *blas_path = default_blas_path();
    if (argc > 1) {
        matrix_path = argv[1];
    }
    if (argc > 2) {
        blas_path = argv[2];
    }

    float *a = aligned_alloc(64, (size_t)M * K * sizeof(float));
    float *c = aligned_alloc(64, (size_t)M * M * sizeof(float));
    float *cref = aligned_alloc(64, (size_t)M * M * sizeof(float));
    if (!a || !c || !cref) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    if (read_matrix(matrix_path, a, (size_t)M * K) != 0) {
        return 2;
    }

    void *handle = NULL;
    ssyrk_fn ssyrk = (ssyrk_fn)load_blas(blas_path, "scipy_cblas_ssyrk64_", &handle);
    sgemm_fn sgemm = (sgemm_fn)dlsym(handle, "scipy_cblas_sgemm64_");
    if (!ssyrk || !sgemm) {
        fprintf(stderr, "missing SSYRK/SGEMM symbols in %s\n", blas_path);
        if (handle) {
            dlclose(handle);
        }
        return 2;
    }

    printf("OpenBLAS repro: float32 SSYRK vs SGEMM for A @ A.T\n");
    printf("  matrix: %s (%d x %d, row-major float32)\n", matrix_path, M, K);
    printf("  BLAS:   %s\n", blas_path);
    if (getenv("OPENBLAS_CORETYPE")) {
        printf("  OPENBLAS_CORETYPE=%s\n", getenv("OPENBLAS_CORETYPE"));
    }

    /* NumPy cblasfuncs.c SYRK path for A @ A.T (C-contiguous A). */
    memset(c, 0, (size_t)M * M * sizeof(float));
    ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
          M, K, 1.0f, a, K, 0.0f, c, M);
    mirror_upper_to_lower(c, M);
    float ssyrk_max = matrix_max(c, M);
    int ssyrk_ok = matches_reference(ssyrk_max);
    printf("\nSSYRK (RowMajor, Upper, NoTrans, n=%d, k=%d, lda=%d, ldc=%d):\n",
           M, K, K, M);
    printf("  max = %.6f  expected ~%.1f  => %s\n",
           ssyrk_max, REFERENCE_MAX, ssyrk_ok ? "OK" : "FAIL");

    /* Control: SGEMM equivalent to A @ A.T (NoTrans, Trans on same A). */
    memset(c, 0, (size_t)M * M * sizeof(float));
    sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          M, M, K, 1.0f, a, K, a, K, 0.0f, c, M);
    float sgemm_max = matrix_max(c, M);
    int sgemm_ok = matches_reference(sgemm_max);
    printf("\nSGEMM control (RowMajor, NoTrans, Trans, m=%d, n=%d, k=%d):\n", M, M, K);
    printf("  max = %.6f  expected ~%.1f  => %s\n",
           sgemm_max, REFERENCE_MAX, sgemm_ok ? "OK" : "FAIL");

    reference_gemm_double(a, cref);
    float ref_max = matrix_max(cref, M);
    printf("\nNaive float64 reference GEMM:\n");
    printf("  max = %.6f\n", ref_max);

    dlclose(handle);
    free(a);
    free(c);
    free(cref);

    if (!ssyrk_ok && sgemm_ok) {
        printf("\nReproduced: SSYRK broken, SGEMM OK (matches NumPy issue).\n");
        return 1;
    }
    if (ssyrk_ok && sgemm_ok) {
        printf("\nBoth paths OK on this host.\n");
        return 0;
    }
    printf("\nUnexpected outcome (SSYRK ok=%d, SGEMM ok=%d).\n", ssyrk_ok, sgemm_ok);
    return ssyrk_ok ? 0 : 1;
}
