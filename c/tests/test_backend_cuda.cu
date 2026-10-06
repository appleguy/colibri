#include "../backend_cuda.h"
#include "../sparse_index.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
/* MSVC has no POSIX setenv/unsetenv */
static int setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
static int unsetenv(const char *name) { return _putenv_s(name, ""); }
#endif

static int close_enough(const float *got, const float *want, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - want[i]) > 1e-4f) {
            std::fprintf(stderr, "mismatch %d: got %.6f want %.6f\n", i, got[i], want[i]);
            return 0;
        }
    }
    return 1;
}

static float hc_ref_sigmoid(float x){
    if(x>=0.f){ float d=::expf(-x); return 1.f/(1.f+d); }
    float g=::expf(x); return g/(1.f+g);
}
static int hc_ref_pre(float *out,float *post,float *comb,const float *input,
                      const float *fn,const float scale[3],const float *base,
                      int hc,int D,int iterations,float norm_eps,float hc_eps){
    if(hc<1||hc>8||D<1||iterations<1) return -1;
    const int flat=hc*D,mix_count=(2+hc)*hc;
    float mixes[80],pre[8],sums[8];
    float ms=0.f;
    for(int i=0;i<flat;i++) ms += input[i]*input[i];
    float inv=1.f/::sqrtf(ms/flat+norm_eps);
    for(int row=0;row<mix_count;row++){
        float sum=0.f;
        for(int col=0;col<flat;col++) sum += fn[(size_t)row*flat+col]*input[col];
        mixes[row]=sum*inv;
    }
    for(int i=0;i<hc;i++){
        pre[i]=hc_ref_sigmoid(mixes[i]*scale[0]+base[i])+hc_eps;
        post[i]=2.f*hc_ref_sigmoid(mixes[hc+i]*scale[1]+base[hc+i]);
    }
    const int mo=2*hc;
    for(int row=0;row<hc;row++){
        float maximum=-INFINITY;
        for(int col=0;col<hc;col++){
            int idx=mo+row*hc+col;
            float v=mixes[idx]*scale[2]+base[idx];
            comb[row*hc+col]=v;
            if(v>maximum) maximum=v;
        }
        float sum=0.f;
        for(int col=0;col<hc;col++){
            float v=::expf(comb[row*hc+col]-maximum);
            comb[row*hc+col]=v; sum+=v;
        }
        for(int col=0;col<hc;col++) comb[row*hc+col]=comb[row*hc+col]/sum+hc_eps;
    }
    for(int col=0;col<hc;col++){
        float sum=0.f;
        for(int row=0;row<hc;row++) sum+=comb[row*hc+col];
        sums[col]=sum;
    }
    for(int row=0;row<hc;row++)
        for(int col=0;col<hc;col++) comb[row*hc+col]/=sums[col]+hc_eps;
    for(int it=1;it<iterations;it++){
        for(int row=0;row<hc;row++){
            float sum=0.f;
            for(int col=0;col<hc;col++) sum+=comb[row*hc+col];
            sums[row]=sum;
        }
        for(int row=0;row<hc;row++)
            for(int col=0;col<hc;col++) comb[row*hc+col]/=sums[row]+hc_eps;
        for(int col=0;col<hc;col++){
            float sum=0.f;
            for(int row=0;row<hc;row++) sum+=comb[row*hc+col];
            sums[col]=sum;
        }
        for(int row=0;row<hc;row++)
            for(int col=0;col<hc;col++) comb[row*hc+col]/=sums[col]+hc_eps;
    }
    for(int col=0;col<D;col++){
        float sum=0.f;
        for(int copy=0;copy<hc;copy++) sum+=pre[copy]*input[(size_t)copy*D+col];
        out[col]=sum;
    }
    return 0;
}

static int relative_rms(const float *got,const float *want,int n,float limit){
    double err=0,ref=0; for(int i=0;i<n;i++){double d=got[i]-want[i];err+=d*d;ref+=(double)want[i]*want[i];}
    float r=(float)std::sqrt(err/(ref+1e-20));
    if(r>limit){std::fprintf(stderr,"relative RMS %.5f exceeds %.5f\n",r,limit);return 0;} return 1;
}

/* ---- fmt=6 (E8/IQ3) --------------------------------------------------------
 * The reference below is written from the format description, not shared with
 * quant.h's decoder, so a common-mode mistake in one cannot hide in the other.
 * A super-block is 98 bytes per 256 weights: 64 codebook indices, then 8 words
 * of (4x7 sign bits + a 4-bit sub-scale in the top nibble), then an fp16 scale.
 * Two indices feed each group of 8 weights; the 8th sign is the parity of the
 * other 7. Scales live in the block, so fmt=6 tensors carry no scale array. */
#define T6_QK 256
#define T6_SUB 32
#define T6_BB  98

static void t6_decode_row(const uint8_t *row, int I, float *w,
                          const uint8_t grid[256][4]) {
    int nb = (I + T6_QK - 1) / T6_QK;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + (size_t)b*T6_BB;
        uint16_t h = (uint16_t)blk[96] | ((uint16_t)blk[97] << 8);
        /* fp16 -> float, written out rather than reusing any helper */
        uint32_t sg=(uint32_t)(h&0x8000)<<16, ex=(h>>10)&0x1F, mn=h&0x3FF, bits;
        if (!ex)         bits = mn ? (sg|((127u-15u)<<23)|(mn<<13)) : sg;
        else if (ex==31) bits = sg|0x7F800000u|(mn<<13);
        else             bits = sg|((ex+112u)<<23)|(mn<<13);
        float d; std::memcpy(&d,&bits,4);
        for (int ib = 0; ib < T6_QK/T6_SUB; ib++) {
            int base = b*T6_QK + ib*T6_SUB;
            if (base >= I) return;
            const uint8_t *wp = blk + 64 + ib*4;
            uint32_t word = (uint32_t)wp[0] | ((uint32_t)wp[1]<<8) |
                            ((uint32_t)wp[2]<<16) | ((uint32_t)wp[3]<<24);
            float db = d * (0.5f + (float)(word >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                uint32_t sev = (word >> (7*l)) & 0x7F;
                const uint8_t *ga = grid[blk[ib*8 + l*2]];
                const uint8_t *gb = grid[blk[ib*8 + l*2 + 1]];
                int parity = 0;
                for (int j = 0; j < 8; j++) {
                    int idx = base + l*8 + j;
                    if (idx >= I) break;
                    int neg;
                    if (j < 7) { neg = (sev >> j) & 1; parity ^= neg; }
                    else       { neg = parity; }
                    float mag = (float)(j < 4 ? ga[j] : gb[j-4]) * 0.5f;
                    w[idx] = neg ? -mag*db : mag*db;
                }
            }
        }
    }
}

/* Sign stream: xorshift64* seeded 417+n, one bit per element (quant.h's
 * e8_signs regenerates the identical stream, and the converter drew it too). */
static void t6_signs(uint8_t *bits, int n) {
    uint64_t s = 417u + (uint64_t)n;
    for (int i = 0; i < (n+7)/8; i++) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        bits[i] = (uint8_t)((s * 2685821657736338717ULL) >> 56);
    }
}
/* y = Q^T x, Q = D*H/sqrt(n); non-power-of-two dims tile block-diagonally. */
static void t6_rot(float *row, int dim) {
    int off = 0;
    while (off < dim) {
        int rem = dim-off, n = rem & (-rem);
        while (n > 4096) n >>= 1;
        uint8_t bits[4096/8]; t6_signs(bits, n);
        float *a = row + off;
        for (int i = 0; i < n; i++) if (bits[i>>3]>>(i&7)&1) a[i] = -a[i];
        for (int len = 1; len < n; len <<= 1)
            for (int i = 0; i < n; i += len<<1)
                for (int j = i; j < i+len; j++) {
                    float u=a[j], v=a[j+len]; a[j]=u+v; a[j+len]=u-v;
                }
        float sc = 1.0f/std::sqrt((float)n);
        for (int i = 0; i < n; i++) a[i] *= sc;
        off += n;
    }
}

static uint32_t t6_rng_state = 0x2545F491u;
static uint32_t t6_rng(void){ t6_rng_state ^= t6_rng_state<<13; t6_rng_state ^= t6_rng_state>>17;
                              t6_rng_state ^= t6_rng_state<<5; return t6_rng_state; }

/* Build a random-but-valid fmt=6 tensor: every index and sign pattern is legal,
 * only the fp16 scale is constrained so the comparison stays meaningful. */
static void t6_fill(uint8_t *q, int I, int O) {
    int nb = (I + T6_QK - 1) / T6_QK;
    for (int o = 0; o < O; o++)
        for (int b = 0; b < nb; b++) {
            uint8_t *blk = q + ((size_t)o*nb + b)*T6_BB;
            for (int i = 0; i < 96; i++) blk[i] = (uint8_t)t6_rng();
            uint16_t h = (uint16_t)((t6_rng() & 0x03FF) | (uint32_t)((10 + t6_rng()%6) << 10));
            blk[96] = (uint8_t)(h & 0xFF); blk[97] = (uint8_t)(h >> 8);
        }
}

static int test_fmt6(int dev) {
    /* A synthetic codebook: the engine passes quant.h's real table, but any 256x4
     * byte table is valid, and a synthetic one keeps this test self-contained
     * while still exercising the upload path the real table travels. */
    static uint8_t grid[256][4];
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 4; j++) grid[i][j] = (uint8_t)((i*4 + j) % 17);
    if (!coli_cuda_e8_set_grid(grid)) { std::fprintf(stderr,"e8 grid upload failed\n"); return 0; }

    const int I = 256, O = 128, S = 2;
    int nb = (I + T6_QK - 1) / T6_QK;
    uint8_t *q = (uint8_t*)std::malloc((size_t)O*nb*T6_BB);
    t6_fill(q, I, O);
    float *x = (float*)std::malloc((size_t)S*I*sizeof(float));
    for (int i = 0; i < S*I; i++) x[i] = std::sin((float)(i+1)*0.031f);

    /* --- matmul --- */
    float *want = (float*)std::calloc((size_t)S*O, sizeof(float));
    float *wrow = (float*)std::malloc((size_t)I*sizeof(float));
    for (int o = 0; o < O; o++) {
        t6_decode_row(q + (size_t)o*nb*T6_BB, I, wrow, grid);
        for (int s = 0; s < S; s++) {
            double acc = 0;
            for (int i = 0; i < I; i++) acc += (double)x[s*I+i]*(double)wrow[i];
            want[s*O+o] = (float)acc;
        }
    }
    float *got = (float*)std::malloc((size_t)S*O*sizeof(float));
    ColiCudaTensor *t6 = nullptr;
    int ok = coli_cuda_matmul(&t6, got, x, q, nullptr, 6, S, I, O, dev, 0);
    if (!ok) { std::fprintf(stderr,"fmt=6 matmul rejected\n"); return 0; }
    if (!relative_rms(got, want, S*O, 1e-4f)) { std::fprintf(stderr,"fmt=6 matmul mismatch\n"); return 0; }

    /* --- expert MLP: gate/up, silu, the device-side down rotation, down ---
     * The caller owns the gate/up input rotation (once per layer), so x goes in
     * as-is; the backend must rotate the silu product before the down matmul. */
    uint8_t *qu = (uint8_t*)std::malloc((size_t)O*nb*T6_BB);
    t6_fill(qu, I, O);
    int nbd = (O + T6_QK - 1) / T6_QK;
    uint8_t *qd = (uint8_t*)std::malloc((size_t)I*nbd*T6_BB);
    t6_fill(qd, O, I);

    float *g = (float*)std::malloc((size_t)S*O*sizeof(float));
    float *u = (float*)std::malloc((size_t)S*O*sizeof(float));
    float *wr2 = (float*)std::malloc((size_t)O*sizeof(float));
    for (int o = 0; o < O; o++) {
        t6_decode_row(q  + (size_t)o*nb*T6_BB, I, wrow, grid);
        float *wu = (float*)std::malloc((size_t)I*sizeof(float));
        t6_decode_row(qu + (size_t)o*nb*T6_BB, I, wu, grid);
        for (int s = 0; s < S; s++) {
            double a = 0, b = 0;
            for (int i = 0; i < I; i++) { a += (double)x[s*I+i]*(double)wrow[i];
                                          b += (double)x[s*I+i]*(double)wu[i]; }
            g[s*O+o] = (float)a; u[s*O+o] = (float)b;
        }
        std::free(wu);
    }
    for (int i = 0; i < S*O; i++) g[i] = (g[i]/(1.0f+std::exp(-g[i]))) * u[i];
    for (int s = 0; s < S; s++) t6_rot(g + (size_t)s*O, O);      /* down input */
    float *want_e = (float*)std::calloc((size_t)S*I, sizeof(float));
    for (int o = 0; o < I; o++) {
        t6_decode_row(qd + (size_t)o*nbd*T6_BB, O, wr2, grid);
        for (int s = 0; s < S; s++) {
            double acc = 0;
            for (int i = 0; i < O; i++) acc += (double)g[s*O+i]*(double)wr2[i];
            want_e[s*I+o] = (float)acc;
        }
    }
    ColiCudaTensor *tg6=nullptr,*tu6=nullptr,*td6=nullptr;
    if (!coli_cuda_tensor_upload(&tg6,q, nullptr,6,I,O,dev) ||
        !coli_cuda_tensor_upload(&tu6,qu,nullptr,6,I,O,dev) ||
        !coli_cuda_tensor_upload(&td6,qd,nullptr,6,O,I,dev)) {
        std::fprintf(stderr,"fmt=6 expert upload failed\n"); return 0;
    }
    float *got_e = (float*)std::malloc((size_t)S*I*sizeof(float));
    if (!coli_cuda_expert_mlp(tg6,tu6,td6,got_e,x,S)) {
        std::fprintf(stderr,"fmt=6 expert_mlp rejected\n"); return 0;
    }
    if (!relative_rms(got_e, want_e, S*I, 2e-4f)) {
        std::fprintf(stderr,"fmt=6 expert_mlp mismatch (device-side rotation?)\n"); return 0;
    }
    ColiCudaTensor *eg6[2]={tg6,tg6},*eu6[2]={tu6,tu6},*ed6[2]={td6,td6};
    int erows[2]={1,1};
    float *group_e=(float*)std::malloc((size_t)S*I*sizeof(float));
    if (!coli_cuda_expert_group(eg6,eu6,ed6,erows,2,group_e,x) ||
        !relative_rms(group_e,want_e,S*I,2e-4f)) {
        std::fprintf(stderr,"fmt=6 expert_group mismatch\n"); return 0;
    }

    coli_cuda_tensor_free(t6); coli_cuda_tensor_free(tg6);
    coli_cuda_tensor_free(tu6); coli_cuda_tensor_free(td6);
    std::free(q); std::free(qu); std::free(qd); std::free(x); std::free(want);
    std::free(got); std::free(wrow); std::free(wr2); std::free(g); std::free(u);
    std::free(want_e); std::free(got_e); std::free(group_e);
    return 1;
}

/* ---- fmt=8 (fp8-e4m3) absorb decode -----------------------------------
 * Exercises weight_at's new fmt=8 branch and absorb_scale's new per-128x128-
 * block branch (this PR pair) through the REAL attention_absorb kernel and
 * absorb_fmt_ok gate, against a CPU reference built the same way the fmt=0
 * absorb block in main() (below) is: independent score/softmax/context
 * accumulation, only the weight lookup itself changes to an e4m3 block-scale
 * dequant. Dims are chosen so BOTH the row-block and column-block axes carry
 * a partial tail block (O=H*(Q+V)=160 -> nblkO=2, rows 128..159 partial;
 * K=140 -> nblkI=2, cols 128..139 partial) -- the exact geometry the new
 * branches must index correctly (blkO=row>>7, blkI=k>>7, scale index
 * blkO*nblkI+blkI). The e4m3 reference decoder is arithmetic (sign/exp/mant,
 * OCP E4M3-FN policy), not the engine's c_e4m3 LUT, so it cross-checks
 * coli_cuda_fp8_set_lut's uploaded table rather than assuming it -- same
 * independence discipline as t8_e4m3_ref's sibling in tests/test_fp8_cuda.cu. */
static float t8_e4m3_ref(uint8_t b) {
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;              /* E4M3-FN: only NaN, no inf */
    float v = e ? ldexpf(1.f + m/8.f, e-7) : ldexpf(m/8.f, -6);
    return s ? -v : v;
}
static uint32_t t8_rng_state = 0xC001D00Du;
static uint8_t t8_rnd_byte(void) {
    t8_rng_state ^= t8_rng_state<<13; t8_rng_state ^= t8_rng_state>>17; t8_rng_state ^= t8_rng_state<<5;
    uint8_t b = (uint8_t)(t8_rng_state & 0xFF);
    if ((b & 0x7F) == 0x7F) b &= (uint8_t)~1;        /* avoid the two NaN byte patterns */
    return b;
}
static float t8_dequant(const uint8_t *q, const float *scale, int I, int row, int col, int nblkI) {
    int blkO = row >> 7, blkI = col >> 7;
    return t8_e4m3_ref(q[(size_t)row*I + col]) * scale[(size_t)blkO*nblkI + blkI];
}

static int test_fmt8_absorb(int dev) {
    float lut[256]; for (int i = 0; i < 256; i++) lut[i] = t8_e4m3_ref((uint8_t)i);
    if (!coli_cuda_fp8_set_lut(lut)) { std::fprintf(stderr,"fmt=8 absorb: set_lut failed\n"); return 0; }

    const int H = 2, Q = 40, V = 40, R = 2, K = 140, T = 3, O = H*(Q+V);
    const int nblkO = (O+127)/128, nblkI = (K+127)/128, nblk = nblkO*nblkI;
    uint8_t *w = (uint8_t*)std::malloc((size_t)O*K);
    for (size_t i = 0; i < (size_t)O*K; i++) w[i] = t8_rnd_byte();
    float *wscale = (float*)std::malloc((size_t)nblk*sizeof(float));
    for (int i = 0; i < nblk; i++) wscale[i] = 0.01f + 0.002f*(float)i;

    /* Refusal must have held BEFORE this test ever ran (fmt=8 was invisible to
     * absorb_fmt_ok on unpatched main) -- upload + launch below is the positive
     * side of the same predicate this PR widened. */
    ColiCudaTensor *wt = nullptr;
    if (!coli_cuda_tensor_upload(&wt, w, wscale, 8, K, O, dev)) {
        std::fprintf(stderr,"fmt=8 absorb: weight upload rejected\n"); return 0;
    }

    float *q = (float*)std::malloc((size_t)H*(Q+R)*sizeof(float));
    float *latent = (float*)std::malloc((size_t)T*K*sizeof(float));
    float *rope = (float*)std::malloc((size_t)T*R*sizeof(float));
    for (int i = 0; i < H*(Q+R); i++) q[i] = std::sin((float)(i+1)*0.037f);
    for (int i = 0; i < T*K; i++) latent[i] = std::sin((float)(i+1)*0.019f)*0.5f;
    for (int i = 0; i < T*R; i++) rope[i] = std::cos((float)(i+1)*0.041f)*0.3f;
    float *ctx = (float*)std::malloc((size_t)H*V*sizeof(float));
    float scale = 1.f/std::sqrt((float)K);

    if (!coli_cuda_attention_absorb(wt, ctx, q, latent, rope, H, Q, R, V, K, T, scale)) {
        std::fprintf(stderr,"fmt=8 absorb: kernel launch rejected (absorb_fmt_ok gate?)\n"); return 0;
    }

    int bad = 0;
    for (int h = 0; h < H; h++) {
        int rbase = h*(Q+V);
        float qa[512];   /* K<=512, this attention_absorb's own documented bound */
        for (int k = 0; k < K; k++) {
            double a = 0;
            for (int d = 0; d < Q; d++) a += (double)q[h*(Q+R)+d]*t8_dequant(w,wscale,K,rbase+d,k,nblkI);
            qa[k] = (float)a;
        }
        float scores[T];
        for (int t = 0; t < T; t++) {
            double a = 0; for (int k = 0; k < K; k++) a += (double)qa[k]*latent[t*K+k];
            for (int d = 0; d < R; d++) a += (double)q[h*(Q+R)+Q+d]*rope[t*R+d];
            scores[t] = (float)a*scale;
        }
        float mx = scores[0]; for (int t = 1; t < T; t++) mx = scores[t]>mx?scores[t]:mx;
        float z = 0; for (int t = 0; t < T; t++) { scores[t] = std::exp(scores[t]-mx); z += scores[t]; }
        for (int t = 0; t < T; t++) scores[t] /= z;
        float cl[512];
        for (int k = 0; k < K; k++) { double a=0; for (int t=0;t<T;t++) a += (double)scores[t]*latent[t*K+k]; cl[k]=(float)a; }
        for (int v = 0; v < V; v++) {
            int row = rbase+Q+v;
            double a = 0; for (int k = 0; k < K; k++) a += (double)cl[k]*t8_dequant(w,wscale,K,row,k,nblkI);
            float want = (float)a, got = ctx[h*V+v];
            float rel = std::fabs(want)>1e-4f ? std::fabs(got-want)/std::fabs(want) : std::fabs(got-want);
            if (rel > 1e-3f) {
                std::fprintf(stderr,"fmt=8 absorb mismatch h=%d v=%d got=%.6f want=%.6f rel=%.4g\n",h,v,got,want,rel);
                bad++;
            }
        }
    }
    coli_cuda_tensor_free(wt);
    std::free(w); std::free(wscale); std::free(q); std::free(latent); std::free(rope); std::free(ctx);
    return bad == 0;
}

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES], ndev = argc > 1 ? argc - 1 : 1;
    if (ndev > COLI_CUDA_MAX_DEVICES) return 2;
    for (int i = 0; i < ndev; i++) devices[i] = argc > 1 ? std::atoi(argv[i + 1]) : 0;
    if (!coli_cuda_init(devices, ndev)) return 77;
    if (coli_cuda_device_count() != ndev) return 1;
    int d0 = devices[0], d1 = devices[ndev > 1 ? 1 : 0];
    size_t count = 99, bytes = 99;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    const float x[8] = {1, -2, 3, -4, 2, 1, -1, 0.5f};
    float got[4];

    const int8_t q8[8] = {1, 2, 3, 4, -1, 2, -3, 4};
    const float s8[2] = {0.5f, 2.0f};
    const float want8[4] = {-5.0f, -60.0f, 1.5f, 10.0f};
    ColiCudaTensor *t8 = nullptr;
    if (!coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d0)) return 1;
    if (coli_cuda_tensor_upload(&t8, q8, s8, 1, 5, 2, d0)) return 1;
    if (ndev > 1 && coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d1)) return 1;
    if (!coli_cuda_matmul(&t8, got, x, q8, s8, 1, 2, 4, 2, d0, 0) || !close_enough(got, want8, 4)) return 1;
    /* Cached tensor must stay callable without live host pointers
     * (CUDA_RELEASE_HOST slots null theirs after upload) — including
     * SUSTAINED reuse, not just the first call. */
    for (int rep = 0; rep < 64; rep++)
        if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
            !close_enough(got, want8, 4)) return 1;
    /* A tensor uploaded from a TEMPORARY host buffer must survive the buffer
     * being scribbled and freed (the release-host lifecycle). */
    {
        int8_t *tmpw = static_cast<int8_t *>(std::malloc(8));
        float  *tmps = static_cast<float *>(std::malloc(2 * sizeof(float)));
        if (!tmpw || !tmps) return 2;
        for (int i = 0; i < 8; i++) tmpw[i] = q8[i];
        tmps[0] = s8[0]; tmps[1] = s8[1];
        ColiCudaTensor *tt = nullptr;
        if (!coli_cuda_tensor_upload(&tt, tmpw, tmps, 1, 4, 2, d0)) return 1;
        for (int i = 0; i < 8; i++) tmpw[i] = 99;
        std::free(tmpw); std::free(tmps);
        if (!coli_cuda_matmul(&tt, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
            !close_enough(got, want8, 4)) return 1;
        coli_cuda_tensor_free(tt);
    }
    /* Upload failures must be graceful and must not corrupt accounting —
     * and must not poison LATER healthy launches (sticky-error regression). */
    {
        size_t c0 = 0, b0 = 0, c1 = 0, b1 = 0;
        coli_cuda_stats(-1, &c0, &b0);
        ColiCudaTensor *bad = nullptr;
        if (coli_cuda_tensor_upload(&bad, q8, s8, 1, 4, 2, 9999)) return 1;
        /* 99, not 7: this line asserted "unknown format must fail" with 7 back
         * when 7 WAS unknown. db0c80f made fmt=7 (MXFP4) real for Kimi K3's
         * CUDA tier and the expectation silently inverted -- v1.6.0's first
         * real-GPU run failed exactly here (#971). Unknown must stay unknown. */
        if (coli_cuda_tensor_upload(&bad, q8, s8, 99, 4, 2, d0)) return 1;
        if (coli_cuda_tensor_upload(&bad, q8, nullptr, 1, 4, 2, d0)) return 1;
        if (coli_cuda_tensor_upload(&bad, nullptr, s8, 1, 4, 2, d0)) return 1;
        /* Deliberate ~16 TB ask: the backend PRINTS its out-of-memory line
         * here and that is the expected outcome, not a swallowed failure --
         * announce it so the stderr does not read like a bug (#971 round 1). */
        std::fprintf(stderr, "expecting one out-of-memory line next (deliberate ~16 TB negative case):\n");
        if (coli_cuda_tensor_upload(&bad, q8, s8, 1, 1 << 20, 1 << 24, d0)) return 1; /* ~16 TB */
        if (bad) return 1;
        coli_cuda_stats(-1, &c1, &b1);
        if (c0 != c1 || b0 != b1) return 1;
        /* fmt=7 is REAL now (db0c80f): a valid MXFP4 upload must succeed.
         * Smoke, not correctness -- e2m1 nibbles 2/byte, one ue8m0 exponent
         * per 32 columns; all-zero payload decodes to zeros, which is enough
         * to prove the format is accepted and accounted. The matmul-level
         * correctness case mirrors fmt=6's harness and is tracked in #971. */
        {
            const int I7 = 64, O7 = 4, NG7 = (I7 + 31) / 32;
            std::vector<uint8_t> w7((size_t)O7 * ((I7 + 1) / 2), 0);
            std::vector<uint8_t> s7((size_t)O7 * NG7, 127); /* e8m0 127 = scale 1.0 */
            ColiCudaTensor *t7 = nullptr;
            if (!coli_cuda_tensor_upload(&t7, w7.data(),
                                         reinterpret_cast<const float *>(s7.data()),
                                         7, I7, O7, d0) || !t7) {
                std::fprintf(stderr, "fmt=7 (MXFP4) upload rejected -- the format is real since db0c80f\n");
                return 1;
            }
            coli_cuda_tensor_free(t7);
        }
        /* healthy launch immediately after the failed allocation */
        if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
            !close_enough(got, want8, 4)) return 1;
    }
    /* Fault injection hook: on/off, restores cleanly. */
    if (setenv("COLI_GPU_FAIL_AFTER", "0", 1)) return 2;
    if (coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0)) return 1;
    if (unsetenv("COLI_GPU_FAIL_AFTER")) return 2;
    if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
        !close_enough(got, want8, 4)) return 1;
    const int8_t q8b[8]={-1,-2,-3,-4, 1,-2,3,-4};
    const float s8b[2]={1.f,.5f},want8b[4]={10.f,15.f,-3.f,-2.5f};
    if(!coli_cuda_tensor_update(t8,q8b,s8b)||
       !coli_cuda_matmul(&t8,got,x,q8b,s8b,1,2,4,2,d0,0)||
       !close_enough(got,want8b,4))return 1;

    /* Rows [-8,-1,0,7] and [1,2,3,4], packed low nibble first. */
    const uint8_t q4[4] = {0x70, 0xf8, 0xa9, 0xcb};
    const float s4[2] = {1.0f, 0.25f};
    const float want4[2] = {-34.0f, -2.5f};
    ColiCudaTensor *t4 = nullptr;
    if (!coli_cuda_matmul(&t4, got, x, q4, s4, 2, 1, 4, 2, d1, 0) || !close_enough(got, want4, 2)) return 1;

    const uint8_t q2[2] = {0xe4, 0x1b};
    const float s2[2] = {0.5f, 2.0f};
    const float want2[2] = {-2.0f, 12.0f};
    ColiCudaTensor *t2 = nullptr;
    if (!coli_cuda_matmul(&t2, got, x, q2, s2, 3, 1, 4, 2, d1, 0) || !close_enough(got, want2, 2)) return 1;

    const float wf[8] = {1, 0, -1, 2, 0.5f, 0.5f, 0.5f, 0.5f};
    const float wantf[2] = {-10.0f, -1.0f};
    ColiCudaTensor *tf = nullptr;
    if (!coli_cuda_matmul(&tf, got, x, wf, nullptr, 0, 1, 4, 2, d0, 0) || !close_enough(got, wantf, 2)) return 1;

    const float eg[8] = {1,0,0,0, 0,1,0,0};
    const float eu[8] = {1,0,0,0, 0,1,0,0};
    const float ed[8] = {1,0, 0,1, 1,1, 1,-1};
    ColiCudaTensor *tg=nullptr,*tu=nullptr,*td=nullptr;
    if (!coli_cuda_tensor_upload_g(&tg,eg,nullptr,0,4,2,d0,0) ||
        !coli_cuda_tensor_upload_g(&tu,eu,nullptr,0,4,2,d0,0) ||
        !coli_cuda_tensor_upload_g(&td,ed,nullptr,0,2,4,d0,0)) return 1;
    float expert[8], want_expert[8];
    for(int s=0;s<2;s++){
        float a=x[s*4], b=x[s*4+1];
        a=(a/(1.0f+std::exp(-a)))*a; b=(b/(1.0f+std::exp(-b)))*b;
        want_expert[s*4]=a; want_expert[s*4+1]=b;
        want_expert[s*4+2]=a+b; want_expert[s*4+3]=a-b;
    }
    if (!coli_cuda_expert_mlp(tg,tu,td,expert,x,2) ||
        !close_enough(expert,want_expert,8)) return 1;
    ColiCudaTensor *gates[2]={tg,tg},*ups[2]={tu,tu},*downs[2]={td,td};
    int group_rows[2]={1,1}; float grouped[8];
    if (!coli_cuda_expert_group(gates,ups,downs,group_rows,2,grouped,x) ||
        !close_enough(grouped,want_expert,8)) return 1;

    const float aw[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    const float aq[4]={1,2,.5f,-.5f},al[12]={1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float ar[6]={1,0, 0,1, 1,1};float actx[2],aref[2];
    ColiCudaTensor *at=nullptr;if(!coli_cuda_tensor_upload_g(&at,aw,nullptr,0,4,4,d0,0))return 1;
    float score[3];for(int t=0;t<3;t++)score[t]=aq[0]*al[t*4]+aq[1]*al[t*4+1]+aq[2]*ar[t*2]+aq[3]*ar[t*2+1];
    float mx=score[0],z=0;for(int t=1;t<3;t++)mx=score[t]>mx?score[t]:mx;
    for(int t=0;t<3;t++){score[t]=std::exp(score[t]-mx);z+=score[t];}for(int t=0;t<3;t++)score[t]/=z;
    for(int v=0;v<2;v++){aref[v]=0;for(int t=0;t<3;t++)aref[v]+=score[t]*al[t*4+2+v];}
    if(!coli_cuda_attention_absorb(at,actx,aq,al,ar,1,2,2,2,4,3,1.f)||
       !close_enough(actx,aref,2))return 1;
    coli_cuda_tensor_free(at);

    /* GLM-5.3 sparse absorbed MLA: q is already projected into latent space
     * and selected[] names the only cache rows allowed into the softmax. */
    constexpr int AS=2, AH=2, AV=3, AK=4, AT=5, AW=3;
    const float avw[AH*AV*AK]={
        1,0,0,0, 0,1,0,0, 0,0,1,0,
        0,0,0,1, 1,1,0,0, 0,1,1,0
    };
    const float aqa[AS*AH*AK]={
        1,.5f,-.25f,.75f, -.5f,1,.25f,.5f,
        .25f,.5f,1,-.5f, 1,-.25f,.5f,.75f
    };
    const float ala[AT*AK]={
        1,0,.5f,-.5f, 0,1,-.25f,.75f, .5f,.5f,1,0,
        -.5f,.25f,.75f,1, .2f,-.4f,.6f,.8f
    };
    const int asel[AS*AW]={0,2,-1, 1,3,4};
    float agpu[AS*AH*AV], acpu[AS*AH*AV];
    ColiCudaTensor *avt=nullptr;
    if(!coli_cuda_tensor_upload_g(&avt,avw,nullptr,0,AK,AH*AV,d0,0))return 1;
    for(int ss=0;ss<AS;ss++)for(int h=0;h<AH;h++){
        float sc[AW],mx=-INFINITY,z=0,pooled[AK]={0,0,0,0};
        for(int i=0;i<AW;i++){
            int at2=asel[ss*AW+i];
            if(at2<0||at2>=AT){sc[i]=-INFINITY;continue;}
            float a=0;for(int k=0;k<AK;k++)a+=aqa[(ss*AH+h)*AK+k]*ala[at2*AK+k];
            sc[i]=a*.5f;mx=sc[i]>mx?sc[i]:mx;
        }
        for(int i=0;i<AW;i++)if(asel[ss*AW+i]>=0){sc[i]=std::exp(sc[i]-mx);z+=sc[i];}
        for(int i=0;i<AW;i++){
            int at2=asel[ss*AW+i];if(at2<0)continue;float ww=sc[i]/z;
            for(int k=0;k<AK;k++)pooled[k]+=ww*ala[at2*AK+k];
        }
        for(int v=0;v<AV;v++){
            float a=0;for(int k=0;k<AK;k++)a+=pooled[k]*avw[(h*AV+v)*AK+k];
            acpu[(ss*AH+h)*AV+v]=a;
        }
    }
    if(!coli_cuda_attention_absorbed_sparse_batch(avt,agpu,aqa,ala,asel,AS,AH,AV,AK,AT,AW,.5f)||
       !close_enough(agpu,acpu,AS*AH*AV)){
        std::fprintf(stderr,"absorbed sparse CUDA attention mismatch\n");
        return 1;
    }
    constexpr int AO=AH*AV;
    float opw[AO*AO]={0}, aproj[AS*AO];
    for(int i=0;i<AO;i++)opw[i*AO+i]=1.f;
    ColiCudaTensor *opt=nullptr;
    if(!coli_cuda_tensor_upload_g(&opt,opw,nullptr,0,AO,AO,d0,0)||
       !coli_cuda_attention_absorbed_sparse_project_batch(
            avt,opt,aproj,aqa,ala,asel,AS,AH,AV,AK,AT,AW,.5f)||
       !close_enough(aproj,agpu,AS*AO)){
        std::fprintf(stderr,"absorbed sparse CUDA attention+projection mismatch\n");
        return 1;
    }
    float adev[AS*AO], aqa_ephemeral[AS*AH*AK];
    int asel_ephemeral[AS*AW];
    std::memcpy(aqa_ephemeral,aqa,sizeof(aqa_ephemeral));
    std::memcpy(asel_ephemeral,asel,sizeof(asel_ephemeral));
    float *adev_out=(float*)coli_cuda_pipe_alloc(d0,sizeof(adev));
    int adev_ok=adev_out &&
       coli_cuda_attention_absorbed_sparse_project_batch_dev_out(
            avt,opt,adev_out,aqa_ephemeral,ala,asel_ephemeral,AS,AH,AV,AK,AT,AW,.5f);
    if(adev_ok){
        /* The dev-out contract may return while MLA/o_proj is still queued.
         * Its ephemeral query/selection inputs must already be safe to reuse. */
        std::memset(aqa_ephemeral,0xa5,sizeof(aqa_ephemeral));
        for(int i=0;i<AS*AW;i++) asel_ephemeral[i]=-1234567;
        adev_ok=coli_cuda_pipe_download(d0,adev_out,adev,sizeof(adev)) &&
                close_enough(adev,aproj,AS*AO);
    }
    if(!adev_ok){
        std::fprintf(stderr,"absorbed sparse CUDA dev-out projection/lifetime mismatch\n");
        return 1;
    }
    coli_cuda_pipe_free(d0,adev_out);

    /* S=1 dev-out compacts only the selected latent rows before H2D. Compare
     * that decode-specialized path against the full-history synchronous path. */
    float aproj1[AO], adev1[AO], aq1[AH*AK];
    int asel1[AW];
    std::memcpy(aq1,aqa+AH*AK,sizeof(aq1));
    std::memcpy(asel1,asel+AW,sizeof(asel1));
    float *adev1_out=(float*)coli_cuda_pipe_alloc(d0,sizeof(adev1));
    if(!adev1_out||
       !coli_cuda_attention_absorbed_sparse_project_batch(
            avt,opt,aproj1,aq1,ala,asel1,1,AH,AV,AK,AT,AW,.5f)||
       !coli_cuda_attention_absorbed_sparse_project_batch_dev_out(
            avt,opt,adev1_out,aq1,ala,asel1,1,AH,AV,AK,AT,AW,.5f)){
        std::fprintf(stderr,"absorbed sparse CUDA compact decode execution failed\n");
        return 1;
    }
    std::memset(aq1,0x5a,sizeof(aq1));
    for(int i=0;i<AW;i++) asel1[i]=-7654321;
    if(!coli_cuda_pipe_download(d0,adev1_out,adev1,sizeof(adev1))||
       !close_enough(adev1,aproj1,AO)){
        std::fprintf(stderr,"absorbed sparse CUDA compact decode mismatch\n");
        return 1;
    }
    coli_cuda_pipe_free(d0,adev1_out);
    coli_cuda_tensor_free(opt);
    coli_cuda_tensor_free(avt);

    /* Device-resident mHC pre: preserve the CPU reduction/Sinkhorn ordering
     * while parallelizing independent tokens, mix rows and output columns. */
    {
        constexpr int HS=2,HC=3,HD=5,HM=(2+HC)*HC;
        float input[HS*HC*HD],fn[HM*HC*HD],scale[3]={.7f,1.1f,.9f},base[HM];
        for(int i=0;i<HS*HC*HD;i++) input[i]=std::sin((i+1)*.13f);
        for(int i=0;i<HM*HC*HD;i++) fn[i]=std::cos((i+3)*.017f)*.2f;
        for(int i=0;i<HM;i++) base[i]=std::sin((i+5)*.11f)*.1f;
        float oref[HS*HD],pref[HS*HC],cref[HS*HC*HC];
        for(int t=0;t<HS;t++){
            if(hc_ref_pre(oref+t*HD,pref+t*HC,cref+t*HC*HC,
                           input+t*HC*HD,fn,scale,base,HC,HD,4,1e-6f,1e-6f)){
                std::fprintf(stderr,"CPU hyperconnection pre reference failed\n");
                return 1;
            }
        }
        float ogot[HS*HD],pgot[HS*HC],cgot[HS*HC*HC];
        float *id=(float*)coli_cuda_pipe_alloc(d0,sizeof(input));
        float *fd=(float*)coli_cuda_pipe_alloc(d0,sizeof(fn));
        float *sd=(float*)coli_cuda_pipe_alloc(d0,sizeof(scale));
        float *bad=(float*)coli_cuda_pipe_alloc(d0,sizeof(base));
        float *od=(float*)coli_cuda_pipe_alloc(d0,sizeof(ogot));
        float *pd=(float*)coli_cuda_pipe_alloc(d0,sizeof(pgot));
        float *cd=(float*)coli_cuda_pipe_alloc(d0,sizeof(cgot));
        if(!id||!fd||!sd||!bad||!od||!pd||!cd||
           !coli_cuda_pipe_upload(d0,id,input,sizeof(input))||
           !coli_cuda_pipe_upload(d0,fd,fn,sizeof(fn))||
           !coli_cuda_pipe_upload(d0,sd,scale,sizeof(scale))||
           !coli_cuda_pipe_upload(d0,bad,base,sizeof(base))||
           !coli_cuda_pipe_hc_pre(d0,od,pd,cd,id,fd,sd,bad,HS,HC,HD,4,1e-6f,1e-6f)||
           !coli_cuda_pipe_download(d0,od,ogot,sizeof(ogot))||
           !coli_cuda_pipe_download(d0,pd,pgot,sizeof(pgot))||
           !coli_cuda_pipe_download(d0,cd,cgot,sizeof(cgot))||
           !close_enough(ogot,oref,HS*HD)||
           !close_enough(pgot,pref,HS*HC)||
           !close_enough(cgot,cref,HS*HC*HC)){
            std::fprintf(stderr,"CUDA hyperconnection pre mismatch\n");
            return 1;
        }
        coli_cuda_pipe_free(d0,cd); coli_cuda_pipe_free(d0,pd); coli_cuda_pipe_free(d0,od);
        coli_cuda_pipe_free(d0,bad); coli_cuda_pipe_free(d0,sd);
        coli_cuda_pipe_free(d0,fd); coli_cuda_pipe_free(d0,id);
    }

    {
        constexpr int HC=3, HD=5;
        float residual[HC*HD]={
            .1f,.2f,.3f,.4f,.5f,
            -.2f,.4f,-.6f,.8f,-1.f,
            1.1f,-1.2f,1.3f,-1.4f,1.5f
        };
        float branch[HD]={.25f,-.5f,.75f,-1.f,1.25f};
        float post[HC]={.5f,-.25f,.75f};
        float comb[HC*HC]={
            .6f,.1f,.3f,
            .2f,.7f,.1f,
            .4f,.2f,.4f
        };
        float href[HC*HD], hgot[HC*HD];
        for(int dst=0;dst<HC;dst++)for(int col=0;col<HD;col++){
            float v=0.f;
            for(int src=0;src<HC;src++)
                v += comb[src*HC+dst]*residual[src*HD+col];
            href[dst*HD+col]=v+post[dst]*branch[col];
        }
        float *rd=(float*)coli_cuda_pipe_alloc(d0,sizeof(residual));
        float *bd=(float*)coli_cuda_pipe_alloc(d0,sizeof(branch));
        float *pd=(float*)coli_cuda_pipe_alloc(d0,sizeof(post));
        float *cd=(float*)coli_cuda_pipe_alloc(d0,sizeof(comb));
        float *od=(float*)coli_cuda_pipe_alloc(d0,sizeof(hgot));
        if(!rd||!bd||!pd||!cd||!od||
           !coli_cuda_pipe_upload(d0,rd,residual,sizeof(residual))||
           !coli_cuda_pipe_upload(d0,bd,branch,sizeof(branch))||
           !coli_cuda_pipe_upload(d0,pd,post,sizeof(post))||
           !coli_cuda_pipe_upload(d0,cd,comb,sizeof(comb))||
           !coli_cuda_pipe_hc_post(d0,od,bd,rd,pd,cd,HC,HD)||
           !coli_cuda_pipe_download(d0,od,hgot,sizeof(hgot))||
           !close_enough(hgot,href,HC*HD)){
            std::fprintf(stderr,"CUDA hyperconnection post mismatch\n");
            return 1;
        }
        coli_cuda_pipe_free(d0,od);
        coli_cuda_pipe_free(d0,cd);
        coli_cuda_pipe_free(d0,pd);
        coli_cuda_pipe_free(d0,bd);
        coli_cuda_pipe_free(d0,rd);
    }

    /* Native s4 WMMA path: compare the quantized-activation result against the
       existing FP32-activation/s4-weight grouped implementation. */
    uint8_t w4[32*32/2]; float ws4[32], gx4[64], scalar4[64], async4[64], tensor4[64], pinned4[64], clamped4[64], resident4[64];
    for(int i=0;i<(int)sizeof(w4);i++){
        int lo=((i%15)-7)&15,hi=(((i*3)%15)-7)&15;
        w4[i]=(uint8_t)(lo|(hi<<4));
    }
    for(int i=0;i<32;i++)ws4[i]=0.01f+(i%5)*0.002f;
    for(int i=0;i<64;i++)gx4[i]=std::sin((float)(i+1)*0.17f)*2.f;
    ColiCudaTensor *g4=nullptr,*u4=nullptr,*d4=nullptr;
    if(!coli_cuda_tensor_upload_g(&g4,w4,ws4,2,32,32,d0,0)||
       !coli_cuda_tensor_upload_g(&u4,w4,ws4,2,32,32,d0,0)||
       !coli_cuda_tensor_upload_g(&d4,w4,ws4,2,32,32,d0,0))return 1;
    ColiCudaTensor *gg4[2]={g4,g4},*ug4[2]={u4,u4},*dg4[2]={d4,d4};
    if(!coli_cuda_expert_group(gg4,ug4,dg4,group_rows,2,scalar4,gx4))return 1;
    if(!coli_cuda_expert_group_issue(gg4,ug4,dg4,group_rows,2,gx4))return 1;
    const float *async_result=coli_cuda_expert_group_take(d0);
    if(!async_result)return 1;
    std::memcpy(async4,async_result,sizeof(async4));
    if(std::memcmp(async4,scalar4,sizeof(async4))){
        std::fprintf(stderr,"async packed-s4 group differs from sync path\n");
        return 1;
    }
    setenv("COLI_CUDA_TC_INT4","1",1);
    setenv("COLI_CUDA_TC_MIN_ROWS","1",1);
    /* #1499: poison the destination first. The device buffer this call writes
     * still held scalar4 from the call above, so an EMPTY tensor-core kernel
     * (rocWMMA, or a card below sm_75 with the body compiled out) left it
     * untouched and the comparison below passed against itself. */
    for(int i=0;i<64;i++)tensor4[i]=-12345.f;
    if(!coli_cuda_expert_group(gg4,ug4,dg4,group_rows,2,tensor4,gx4)||
       !relative_rms(tensor4,scalar4,64,0.30f))return 1;
    if(!coli_cuda_expert_group_pinned(gg4,ug4,dg4,group_rows,2,pinned4,gx4,1)||
       std::memcmp(pinned4,scalar4,sizeof(pinned4))){
        std::fprintf(stderr,"pinned CUDA group did not bypass W4A4 Tensor Cores\n");
        return 1;
    }
    unsetenv("COLI_CUDA_TC_INT4");
    unsetenv("COLI_CUDA_TC_MIN_ROWS");
    setenv("COLI_CUDA_TC_W4A16","1",1);
    setenv("COLI_CUDA_TC_W4A16_MIN","1",1);
    if(!coli_cuda_expert_group_pinned(gg4,ug4,dg4,group_rows,2,pinned4,gx4,1)||
       std::memcmp(pinned4,scalar4,sizeof(pinned4))){
        std::fprintf(stderr,"pinned CUDA group did not bypass W4A16 Tensor Cores\n");
        return 1;
    }
    unsetenv("COLI_CUDA_TC_W4A16");
    unsetenv("COLI_CUDA_TC_W4A16_MIN");
    /* GLM-5.3's clamped SwiGLU path requires grouped-int4 (fmt=4).
     * Compare the generic lazy-upload barrier path with the resident path on
     * already-uploaded tensors: skipping that barrier must be bit-identical. */
    float gs4[64];
    for(int o=0;o<32;o++){ gs4[o*2]=ws4[o]; gs4[o*2+1]=ws4[o]*0.75f; }
    ColiCudaTensor *cg4=nullptr,*cu4=nullptr,*cd4=nullptr;
    if(!coli_cuda_tensor_upload_g(&cg4,w4,gs4,4,32,32,d0,16) ||
       !coli_cuda_tensor_upload_g(&cu4,w4,gs4,4,32,32,d0,16) ||
       !coli_cuda_tensor_upload_g(&cd4,w4,gs4,4,32,32,d0,16)) return 1;
    ColiCudaTensor *cgg4[2]={cg4,cg4},*cug4[2]={cu4,cu4},*cdg4[2]={cd4,cd4};
    if(!coli_cuda_expert_group_clamped(cgg4,cug4,cdg4,group_rows,2,clamped4,gx4,6.f) ||
       !coli_cuda_expert_group_clamped_resident(cgg4,cug4,cdg4,group_rows,2,resident4,gx4,6.f) ||
       std::memcmp(clamped4,resident4,sizeof(clamped4))){
        std::fprintf(stderr,"resident clamped CUDA group differs from generic clamped path\n");
        return 1;
    }
    /* Decode-style async resident issue/take must support GLM-5.3 grouped
     * int4 + clamped SwiGLU too. One expert, one row, gate=1 makes the
     * weighted reduction exactly the synchronous resident reference. */
    {
        ColiCudaTensor *ag[1]={cg4}, *au[1]={cu4}, *ad[1]={cd4};
        int one_row[1]={1}, one_dev[1]={d0};
        float one_ref[32], one_got[32], one_weight[1]={1.f};
        if(!coli_cuda_expert_group_clamped_resident(
                ag,au,ad,one_row,1,one_ref,gx4,6.f)) return 1;
        float *x_dev=(float*)coli_cuda_pipe_alloc(d0,32*sizeof(float));
        float *slot_dev=(float*)coli_cuda_pipe_alloc(d0,32*sizeof(float));
        float *acc_dev=(float*)coli_cuda_pipe_alloc(d0,32*sizeof(float));
        if(!x_dev||!slot_dev||!acc_dev ||
           !coli_cuda_pipe_upload(d0,x_dev,gx4,32*sizeof(float)) ||
           !coli_cuda_expert_group_resident_issue_clamped(
                ag,au,ad,one_weight,1,d0,x_dev,slot_dev,6.f) ||
           !coli_cuda_expert_group_resident_take(
                d0,one_dev,1,slot_dev,acc_dev,32) ||
           !coli_cuda_pipe_download(d0,acc_dev,one_got,32*sizeof(float)) ||
           std::memcmp(one_ref,one_got,sizeof(one_ref))){
            std::fprintf(stderr,"async resident clamped issue/take differs from sync path\n");
            return 1;
        }
        coli_cuda_pipe_free(d0,x_dev);
        coli_cuda_pipe_free(d0,slot_dev);
        coli_cuda_pipe_free(d0,acc_dev);
    }
    coli_cuda_tensor_free(cg4);coli_cuda_tensor_free(cu4);coli_cuda_tensor_free(cd4);
    coli_cuda_tensor_free(g4);coli_cuda_tensor_free(u4);coli_cuda_tensor_free(d4);

    /* GLM-5.3 pipe SwiGLU must match its asymmetric clamp contract:
     * positive-only gate cap, symmetric up cap, then SiLU(gate)*up. */
    {
        constexpr int N=6;
        float gate[N]={10.f,-10.f,2.f,-2.f,0.25f,4.f};
        float up[N]={10.f,-10.f,.5f,-.5f,8.f,-8.f};
        float want[N];
        const float limit=3.f;
        for(int i=0;i<N;i++){
            float g=gate[i]>limit?limit:gate[i];
            float u=up[i]<-limit?-limit:(up[i]>limit?limit:up[i]);
            want[i]=(g/(1.f+std::exp(-g)))*u;
        }
        float got[N];
        float *gd=(float*)coli_cuda_pipe_alloc(d0,sizeof(gate));
        float *ud=(float*)coli_cuda_pipe_alloc(d0,sizeof(up));
        if(!gd||!ud||
           !coli_cuda_pipe_upload(d0,gd,gate,sizeof(gate))||
           !coli_cuda_pipe_upload(d0,ud,up,sizeof(up))||
           !coli_cuda_pipe_swiglu_clamped(d0,gd,ud,N,limit)||
           !coli_cuda_pipe_download(d0,gd,got,sizeof(got))||
           !close_enough(got,want,N)){
            std::fprintf(stderr,"CUDA clamped pipe SwiGLU mismatch\n");
            return 1;
        }
        coli_cuda_pipe_free(d0,ud);
        coli_cuda_pipe_free(d0,gd);
    }

    /* GLM-5.3 decode router contract: correction bias changes selection, while
     * the normalized route weights come from the raw sigmoid logits. */
    {
        constexpr int D=4,E=4,K=2;
        float rx[D]={1.f,2.f,3.f,4.f};
        float rw[E*D]={
            1.f,0.f,0.f,0.f,
            0.f,1.f,0.f,0.f,
            0.f,0.f,1.f,0.f,
            0.f,0.f,0.f,1.f
        };
        float rb[E]={0.5f,0.f,0.f,0.f};
        float *xd=(float*)coli_cuda_pipe_alloc(d0,sizeof(rx));
        float *wd=(float*)coli_cuda_pipe_alloc(d0,sizeof(rw));
        float *bd=(float*)coli_cuda_pipe_alloc(d0,sizeof(rb));
        int idx[K]={-1,-1},keff=0; float ww[K]={0,0};
        const float l0=1.f/(1.f+std::exp(-1.f));
        const float l3=1.f/(1.f+std::exp(-4.f));
        const float z=l0+l3+1e-20f;
        if(!xd||!wd||!bd||
           !coli_cuda_pipe_upload(d0,xd,rx,sizeof(rx))||
           !coli_cuda_pipe_upload(d0,wd,rw,sizeof(rw))||
           !coli_cuda_pipe_upload(d0,bd,rb,sizeof(rb))||
           !coli_cuda_pipe_router(d0,xd,wd,bd,D,E,K,0.f,1,2.5f,idx,ww,&keff)||
           keff!=K||idx[0]!=0||idx[1]!=3||
           std::fabs(ww[0]-l0/z*2.5f)>1e-5f||
           std::fabs(ww[1]-l3/z*2.5f)>1e-5f){
            std::fprintf(stderr,"CUDA GLM router contract mismatch\n");
            return 1;
        }
        coli_cuda_pipe_free(d0,bd);
        coli_cuda_pipe_free(d0,wd);
        coli_cuda_pipe_free(d0,xd);
    }

    /* GLM-5.3 decode sparse-index contract: compare against the exact CPU
     * reference with leading padding, complete pools and an incomplete tail. */
    {
        constexpr int SQ=10,SH=2,SD=4,SP=2,ST=4,SW=ST+SP-1,Q=8;
        float query[SH*SD]={1.f,.25f,-.5f,.75f, -.25f,1.f,.5f,.1f};
        float keys[SQ*SD],gates[SQ*SD],ape[SP*SD]={
            .05f,-.10f,.15f,.02f, -.03f,.08f,-.04f,.12f
        };
        float hw[SH]={.8f,.35f};
        unsigned char valid[SQ]={0,0,1,1,1,1,1,1,1,0};
        for(int t=0;t<SQ;t++)for(int j=0;j<SD;j++){
            keys[t*SD+j]=(float)((t+1)*(j+2))*.03125f +
                         ((t+j)&1 ? .07f : -.02f);
            gates[t*SD+j]=(float)((t*3+j*5)%11)*.07f-.25f;
        }
        int cpu[SW],gpu[SW],cpu_first[SW],gpu_first[SW];
        constexpr int SQ_FIRST=8,Q_FIRST=7;
        if(coli_sparse_index_select_range(cpu_first,query,keys,gates,hw,ape,valid,
                                          SQ_FIRST,SH,SD,SP,ST,1,Q_FIRST,Q_FIRST+1) ||
           !coli_cuda_sparse_index_select_decode(d0,gpu_first,query,keys,gates,hw,ape,
                                                 valid,SQ_FIRST,SH,SD,SP,ST,1,Q_FIRST)){
            std::fprintf(stderr,"GLM sparse index initial cache execution failed\n");
            return 1;
        }
        for(int i=0;i<SW;i++) if(cpu_first[i]!=gpu_first[i]){
            std::fprintf(stderr,
                "GLM sparse index initial cache mismatch at %d: gpu=%d cpu=%d\n",
                i,gpu_first[i],cpu_first[i]);
            return 1;
        }
        if(coli_sparse_index_select_range(cpu,query,keys,gates,hw,ape,valid,
                                          SQ,SH,SD,SP,ST,1,Q,Q+1) ||
           !coli_cuda_sparse_index_select_decode(d0,gpu,query,keys,gates,hw,ape,
                                                 valid,SQ,SH,SD,SP,ST,1,Q)){
            std::fprintf(stderr,"GLM sparse index append execution failed\n");
            return 1;
        }
        for(int i=0;i<SW;i++) if(cpu[i]!=gpu[i]){
            std::fprintf(stderr,
                "GLM sparse index append mismatch at %d: gpu=%d cpu=%d\n",
                i,gpu[i],cpu[i]);
            return 1;
        }
    }

    uint64_t group_calls=0,group_experts=0,group_total_rows=0;
    coli_cuda_group_stats(&group_calls,&group_experts,&group_total_rows,nullptr,nullptr,nullptr);
    if(group_calls!=9||group_experts!=17||group_total_rows!=17) return 1;

    coli_cuda_stats(-1, &count, &bytes);
    if (count != 7 || bytes != 166) {
        std::fprintf(stderr, "unexpected CUDA stats: %zu tensors, %zu bytes\n", count, bytes);
        return 1;
    }
    if (coli_cuda_tensor_device(t8) != d0 || coli_cuda_tensor_device(tf) != d0 ||
        coli_cuda_tensor_device(t4) != d1 || coli_cuda_tensor_device(t2) != d1) return 1;
    coli_cuda_stats(d0, &count, &bytes);
    if (ndev > 1) {
        if (count != 5 || bytes != 144) return 1;
        coli_cuda_stats(d1, &count, &bytes);
        if (count != 2 || bytes != 22) return 1;
    } else if (count != 7 || bytes != 166) return 1;

    coli_cuda_tensor_free(t8);
    coli_cuda_tensor_free(t4);
    coli_cuda_tensor_free(t2);
    coli_cuda_tensor_free(tf);
    coli_cuda_tensor_free(tg);
    coli_cuda_tensor_free(tu);
    coli_cuda_tensor_free(td);
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;

    /* fmt=6 runs after the stats assertions above, which pin exact tensor counts. */
    if (!test_fmt6(d0)) return 1;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) { std::fprintf(stderr,"fmt=6 leaked tensors\n"); return 1; }

    /* fmt=8 absorb: same self-contained lifecycle discipline as fmt=6 above,
     * BOTH halves. The byte half is load-bearing history: an earlier vintage of
     * this feature found coli_cuda_tensor_free subtracting per-row scale bytes
     * for a per-block-scaled fmt=8 tensor (upload charged scale_count =
     * ceil(O/128)*ng, free subtracted O*ng), so the `tensor_bytes >= bytes`
     * guard silently declined the subtraction and the diagnostic VRAM counter
     * stuck non-zero forever after freeing ANY fmt=8 tensor. free's accounting
     * now mirrors upload's charge expression exactly (see the comment in
     * coli_cuda_tensor_free), and this is the assertion that keeps the two
     * from drifting apart again for a tracked fmt=8 tensor. */
    if (!test_fmt8_absorb(d0)) return 1;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) { std::fprintf(stderr,"fmt=8 absorb leaked tensors\n"); return 1; }

    coli_cuda_shutdown();
    std::printf("cuda backend: q8/q4/q2/f32/e8 correctness ok on %d device(s)\n", ndev);
    return 0;
}
