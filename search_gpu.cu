#include <ctype.h>
/* =============================================================================
   search_gpu.cu  --  secp256k1 nearest-prefix search, CUDA + CPU fallback.

   Build on a CUDA machine (RTX 5090 = sm_120, needs CUDA 12.8+):
       nvcc -O3 -arch=sm_120 -o search_gpu search_gpu.cu

   Build here (no GPU) as a CPU/OpenMP fallback, which is what this sandbox
   tested end-to-end against an independent brute-force scan:
       gcc -O3 -fopenmp -DHALF=128 -o search_gpu search_gpu.cu -lm -x c++ -lstdc++

   Usage mirrors p.py:
       search_gpu <pubkey> <M> [gap] [options]
       search_gpu <pubkey> --prefix=<hex> [gap] [options]
   Options: --up / --dir=up|down, --xonly / --ignore_02_03, --count=N,
            --skip=N, --threads=T, --max_n=N, --k=<hex scalar>, --gap=<int>
   HALF (group size) is compile-time: recompile with -DHALF=N (default 128).
============================================================================= */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "fe_core.h"
#include "secp_host.h"
#include "group_logic.h"
#include "kernel_body.h"

#ifdef __CUDACC__
  #define ATOMIC_INC(ptr) atomicAdd((ptr), 1u)
#else
  #ifdef _OPENMP
  #include <omp.h>
  #endif
  static inline uint32_t atomic_inc_cpu(uint32_t *p){
      uint32_t old;
      #pragma omp atomic capture
      { old = *p; *p += 1; }
      return old;
  }
  #define ATOMIC_INC(ptr) atomic_inc_cpu(ptr)
#endif

/* ---- the one place the per-thread search loop is defined --------------- */
#ifdef __CUDACC__
__global__ void search_kernel(const fe *cx_in, const fe *cy_in, fe *cx_out, fe *cy_out,
                               const fe *tabx, const fe *taby, fe bigx, fe bigy,
                               prefmatch pm, uint32_t T, uint64_t iter_base, uint32_t iters,
                               int up, uint32_t *hit_count, uint64_t *hit_dist, uint32_t max_hits)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint32_t)tid >= T) return;
    fe cx = cx_in[tid], cy = cy_in[tid];
    KSEARCH_BODY(cx, cy, tabx, taby, bigx, bigy, (&pm), T, iter_base, iters, up,
                 hit_count, hit_dist, max_hits);
    cx_out[tid] = cx; cy_out[tid] = cy;
}
#else
/* CPU fallback: one call per thread id, OpenMP-parallel over tid in the caller */
static void cpu_thread_step(int tid, fe *cx_io, fe *cy_io, const fe *tabx, const fe *taby,
                             fe bigx, fe bigy, const prefmatch *pm, uint32_t T,
                             uint64_t iter_base, uint32_t iters, int up,
                             uint32_t *hit_count, uint64_t *hit_dist, uint32_t max_hits)
{
    fe cx = cx_io[tid], cy = cy_io[tid];
    KSEARCH_BODY(cx, cy, tabx, taby, bigx, bigy, pm, T, iter_base, iters, up,
                 hit_count, hit_dist, max_hits);
    cx_io[tid] = cx; cy_io[tid] = cy;
}
#endif

/* ---- command-line / orchestration ---------------------------------------- */
typedef struct { uint64_t v[8]; } found_item; /* unused placeholder */

static int has_flag(int argc, char **argv, const char *name){
    for(int i=1;i<argc;i++) if(argv[i][0]=='-'&&argv[i][1]=='-'&&strcmp(argv[i]+2,name)==0) return 1;
    return 0;
}
static const char* get_opt(int argc, char **argv, const char *name){
    size_t ln=strlen(name);
    for(int i=1;i<argc;i++){
        if(argv[i][0]=='-'&&argv[i][1]=='-'&&strncmp(argv[i]+2,name,ln)==0 && argv[i][2+ln]=='='){
            return argv[i]+2+ln+1;
        }
    }
    return NULL;
}

static pt signed_mul(pt G, uint64_t scalar, int sign){
    pt m = mul_pt(G, scalar);
    if(sign>0) return m;
    pt r; r.x=m.x; r.y=fe_sub(fe_zero_v(), m.y); r.inf=m.inf;
    return r;
}

/* found[] bookkeeping shared by phase1/phase2/custom-prefix searches */
typedef struct {
    uint64_t *list; int n, cap;
} foundlist;
static int fl_has(foundlist *f, uint64_t v){ for(int i=0;i<f->n;i++) if(f->list[i]==v) return 1; return 0; }

/* run one search: P0 -/+ m*gap*G for m=1,2,...; collects up to `count` hits (m>skip)
   whose compressed pubkey matches `prefix` (xonly optional). Returns number found. */
static int run_search(pt P0, const char *prefix, int xonly, uint64_t gap, int up,
                       uint32_t T, int count, uint64_t skip, uint64_t max_n,
                       int have_k, u256 kval, const char *label, foundlist *out)
{
    printf("\n[%s] %s, gap = %llu, prefix = %s%s, threads = %u  (initialising start points...)\n",
           label, up?"UP":"DOWN", (unsigned long long)gap, prefix, xonly?" (x only, 02/03 ignored)":"", T);
    fflush(stdout);
    prefmatch pm; pref_to_mask(prefix, xonly, &pm);
    pt G; G.x=fe_from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    { u256 u=u256_from_hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"); memcpy(G.y.v,u.v,sizeof u.v); }
    G.inf=0;
    pt S = mul_pt(G, gap);
    const pt P0_orig = P0;      /* verification must always measure from the true start point,
                                    never from the shifted P0 used after the tiny-start CPU scan */

    time_t t0 = time(NULL);
    #define REPORT(v) do{ \
        if((v) > skip && !fl_has(out,(v))){ \
            pt Q = signed_mul(G, (v)*gap, up?1:-1); Q = aff_add(P0_orig, Q); \
            char pub[67]; hexpub(Q, pub); \
            const char *cmp = xonly ? pub+2 : pub; \
            int ok = strncmp(cmp, prefix, strlen(prefix))==0; \
            out->list[out->n++] = (v); \
            printf("\n[%s] HIT %d: n = %llu   distance = %llu x %llu = %llu   verify=%s\n", \
                   label, out->n, (unsigned long long)(v), (unsigned long long)(v), \
                   (unsigned long long)gap, (unsigned long long)((v)*gap), ok?"OK":"FAILED"); \
            printf("  pubkey = %s\n", pub); \
            if(have_k){ u256 sc = scalar_add_signed(kval, (v), gap, up?1:-1); char sh[65]; u256_to_hex(sc,sh); \
                        printf("  scalar = %s\n", sh); } \
            if(out->n >= count){ printf("[%s] done in %.1fs\n", label, difftime(time(NULL),t0)); return out->n; } \
        } \
    } while(0)

    /* ---- tiny-start handling: if P0 is a small multiple of gap*G, check those points on the CPU ---- */
    uint64_t n0 = 0;
    int64_t sm = small_multiple(P0, S, (int64_t)2*T + 64);
    if(sm != 0){
        int64_t dirn = up ? 1 : -1;
        n0 = (uint64_t)(llabs(sm) + HALF + 2);
        while ((sm + dirn*(int64_t)(n0+HALF+1)) % GS == 0) n0++;
        printf("  start is a tiny multiple (%lld) of gap*G: checking the first %llu points on the CPU...\n",
               (long long)sm, (unsigned long long)n0);
        pt step = up ? S : signed_mul(G, gap, -1);
        pt *pts = block_batch(P0, step, (int)n0);
        for(uint64_t i=1;i<=n0;i++){
            pt Q = pts[i-1];
            if(Q.inf) continue;
            char pub[67]; hexpub(Q,pub);
            const char *cmp = xonly ? pub+2 : pub;
            if(strncmp(cmp, prefix, strlen(prefix))==0) REPORT(i);
        }
        P0 = pts[n0-1];
        free(pts);
    }

    /* ---- build tab[1..HALF] and T centres, plus the signed big step ---- */
    fe *tabx=(fe*)malloc(sizeof(fe)*HALF), *taby=(fe*)malloc(sizeof(fe)*HALF);
    pt *tabrest = block_batch(S, S, HALF-1);
    tabx[0]=S.x; taby[0]=S.y;
    for(int i=0;i<HALF-1;i++){ tabx[i+1]=tabrest[i].x; taby[i+1]=tabrest[i].y; }
    free(tabrest);

    pt stepC0 = signed_mul(G, (uint64_t)(HALF+1)*gap, up?1:-1);
    pt C0 = aff_add(P0, stepC0);
    pt stepGS = signed_mul(G, (uint64_t)GS*gap, up?1:-1);
    fe *cx = (fe*)malloc(sizeof(fe)*T), *cy = (fe*)malloc(sizeof(fe)*T);
    if(T>1){
        pt *centres = block_batch(C0, stepGS, T-1);
        cx[0]=C0.x; cy[0]=C0.y;
        for(uint32_t i=0;i<T-1;i++){ cx[i+1]=centres[i].x; cy[i+1]=centres[i].y; }
        free(centres);
    } else { cx[0]=C0.x; cy[0]=C0.y; }
    pt big = signed_mul(G, (uint64_t)T*GS*gap, up?1:-1);

    uint32_t MAXH = 65536;
    uint32_t hit_count = 0;
    uint64_t *hit_dist = (uint64_t*)malloc(sizeof(uint64_t)*MAXH);

#ifdef __CUDACC__
    fe *d_cxa,*d_cya,*d_cxb,*d_cyb,*d_tabx,*d_taby; uint32_t *d_hc; uint64_t *d_hd;
    cudaMalloc(&d_cxa,T*sizeof(fe)); cudaMalloc(&d_cya,T*sizeof(fe));
    cudaMalloc(&d_cxb,T*sizeof(fe)); cudaMalloc(&d_cyb,T*sizeof(fe));
    cudaMalloc(&d_tabx,HALF*sizeof(fe)); cudaMalloc(&d_taby,HALF*sizeof(fe));
    cudaMalloc(&d_hc,sizeof(uint32_t)); cudaMalloc(&d_hd,MAXH*sizeof(uint64_t));
    cudaMemcpy(d_cxa,cx,T*sizeof(fe),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cya,cy,T*sizeof(fe),cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabx,tabx,HALF*sizeof(fe),cudaMemcpyHostToDevice);
    cudaMemcpy(d_taby,taby,HALF*sizeof(fe),cudaMemcpyHostToDevice);
#endif

    uint32_t iters = 1; uint64_t iter_base = 0; double t_start = (double)clock()/CLOCKS_PER_SEC;
    uint64_t total_pts = 0; time_t tlast = time(NULL);
    while(1){
        hit_count = 0;
#ifdef __CUDACC__
        cudaMemcpy(d_hc,&hit_count,sizeof(uint32_t),cudaMemcpyHostToDevice);
        int blk=256, grid=(T+blk-1)/blk;
        clock_t ts=clock();
        search_kernel<<<grid,blk>>>(d_cxa,d_cya,d_cxb,d_cyb,d_tabx,d_taby,big.x,big.y,
                                     pm,T,iter_base,iters,up,d_hc,d_hd,MAXH);
        cudaDeviceSynchronize();
        double dt=(double)(clock()-ts)/CLOCKS_PER_SEC;
        cudaMemcpy(&hit_count,d_hc,sizeof(uint32_t),cudaMemcpyDeviceToHost);
#else
        clock_t ts=clock();
        #pragma omp parallel for schedule(static)
        for(int tid=0; tid<(int)T; tid++)
            cpu_thread_step(tid, cx, cy, tabx, taby, big.x, big.y, &pm, T, iter_base, iters, up,
                             &hit_count, hit_dist, MAXH);
        double dt=(double)(clock()-ts)/CLOCKS_PER_SEC;
#endif
        uint64_t hi_n = n0 + (uint64_t)T*GS*(iter_base+iters);
        if(hit_count){
            if(hit_count>MAXH) printf("\nwarning: %u hits in one launch, only %u kept\n",hit_count,MAXH);
#ifdef __CUDACC__
            cudaMemcpy(hit_dist,d_hd,sizeof(uint64_t)*MAXH,cudaMemcpyDeviceToHost);
#endif
            uint32_t nn = hit_count<MAXH?hit_count:MAXH;
            /* sort ascending (nn is small; simple insertion sort) */
            for(uint32_t i=1;i<nn;i++){ uint64_t key=hit_dist[i]; int j=i-1;
                while(j>=0 && hit_dist[j]>key){ hit_dist[j+1]=hit_dist[j]; j--; } hit_dist[j+1]=key; }
            for(uint32_t i=0;i<nn;i++){ uint64_t v = hit_dist[i] + n0; REPORT(v); }
        }
#ifdef __CUDACC__
        { fe*t; t=d_cxa; d_cxa=d_cxb; d_cxb=t; t=d_cya; d_cya=d_cyb; d_cyb=t; }
#endif
        iter_base += iters;
        total_pts += (uint64_t)T*GS*iters;
        if(max_n && hi_n>=max_n){ printf("\n[%s] reached n = %llu without enough hits (found: %d)\n",
                                          label,(unsigned long long)hi_n,out->n); break; }
        if(dt<0.3) iters = iters*2 < (1u<<20) ? iters*2 : (1u<<20);
        else if(dt>1.5 && iters>1) iters/=2;
        time_t now=time(NULL);
        if(now-tlast>=2){ tlast=now; double el=(double)(now)-(double)t0;
            printf("\r[%s] n up to %llu   %.1f Mpts/s   hits: %d",
                   label, (unsigned long long)hi_n, el>0? total_pts/el/1e6:0.0, out->n);
            fflush(stdout); }
    }
    #undef REPORT
    free(tabx); free(taby); free(cx); free(cy); free(hit_dist);
#ifdef __CUDACC__
    cudaFree(d_cxa); cudaFree(d_cya); cudaFree(d_cxb); cudaFree(d_cyb);
    cudaFree(d_tabx); cudaFree(d_taby); cudaFree(d_hc); cudaFree(d_hd);
#endif
    return out->n;
}

int main(int argc, char **argv){
    /* collect positional (non --) args */
    char *pos[8]; int npos=0;
    for(int i=1;i<argc;i++) if(!(argv[i][0]=='-'&&argv[i][1]=='-')) pos[npos++]=argv[i];

    int up = has_flag(argc,argv,"up");
    const char *dir = get_opt(argc,argv,"dir");
    if(dir){ if(!strcmp(dir,"up")) up=1; else if(strcmp(dir,"down")) { fprintf(stderr,"--dir must be up or down\n"); return 1; } }
    int xonly = has_flag(argc,argv,"xonly") || has_flag(argc,argv,"ignore_02_03");
    const char *prefix_opt = get_opt(argc,argv,"prefix");
    int custom = prefix_opt != NULL;

    if(npos < 1){ fprintf(stderr,"usage: %s <pubkey> <M>|--prefix=<hex> [gap] [options]\n",argv[0]); return 1; }
    char h[80]; strncpy(h,pos[0],sizeof h); h[sizeof h-1]=0;
    for(char *p=h; *p; p++) *p = (char)tolower((unsigned char)*p);
    if(strlen(h)!=66 || !((h[0]=='0')&&(h[1]=='2'||h[1]=='3'))){
        fprintf(stderr,"pubkey must be a 66-hex-char compressed key\n"); return 1;
    }

    char pref[80]; int M=0;
    const char *gap_pos = NULL;
    if(custom){
        strncpy(pref,prefix_opt,sizeof pref); pref[sizeof pref-1]=0;
        for(char *p=pref; *p; p++) *p=(char)tolower((unsigned char)*p);
        int n=(int)strlen(pref);
        if(xonly){ if(n<1||n>64){ fprintf(stderr,"with --xonly the prefix must be 1-64 hex chars\n"); return 1; } }
        else if(n<1||n>66||(n>=2&&!(pref[0]=='0'&&(pref[1]=='2'||pref[1]=='3')))||(n==1&&pref[0]!='0')){
            fprintf(stderr,"prefix must be 1-66 hex chars and start with 02 or 03 (or use --xonly)\n"); return 1;
        }
        if(npos>=2) gap_pos = pos[1];
    } else {
        if(npos<2){ fprintf(stderr,"need <M> (or use --prefix=)\n"); return 1; }
        M = atoi(pos[1]);
        int maxM = xonly?64:66;
        if(M<1||M>maxM){ fprintf(stderr,"M must be between 1 and %d\n",maxM); return 1; }
        if(xonly) { strncpy(pref,h+2,M); pref[M]=0; } else { strncpy(pref,h,M); pref[M]=0; }
        if(npos>=3) gap_pos = pos[2];
    }

    const char *gap_opt = get_opt(argc,argv,"gap");
    if(gap_pos) gap_opt = gap_pos;
    int count = 1; { const char *o=get_opt(argc,argv,"count"); if(o) count=atoi(o); }
    if(count<1){ fprintf(stderr,"count must be >= 1\n"); return 1; }
    uint64_t max_n = 0; { const char *o=get_opt(argc,argv,"max_n"); if(o) max_n=strtoull(o,0,10); }
    int have_k = 0; u256 kval; memset(&kval,0,sizeof kval);
    { const char *o=get_opt(argc,argv,"k"); if(o){ have_k=1; char buf[65]; memset(buf,'0',64); buf[64]=0;
        int ln=(int)strlen(o); for(int i=0;i<ln;i++) buf[64-ln+i]=o[i]; kval=u256_from_hex(buf); } }
    uint32_t T; { const char *o=get_opt(argc,argv,"threads");
#ifdef _OPENMP
        int def = omp_get_max_threads()*4; if(def<64) def=64;
#else
        int def = 1024;
#endif
        T = o ? (uint32_t)strtoul(o,0,10) : (uint32_t)def; }

    pt P0; if(!decompress(h,&P0)){ fprintf(stderr,"invalid pubkey\n"); return 1; }

    foundlist found; found.cap = count; found.n=0; found.list=(uint64_t*)malloc(sizeof(uint64_t)*count);
    uint64_t gap; uint64_t skip;
    if(custom || gap_opt){
        gap = gap_opt ? strtoull(gap_opt,0,10) : 1;
        if(gap<1){ fprintf(stderr,"gap must be >= 1\n"); return 1; }
        skip = 0; { const char *o=get_opt(argc,argv,"skip"); if(o) skip=strtoull(o,0,10); }
        printf("%s, gap = %llu\n", custom?"Custom prefix mode":"Phase 1 skipped", (unsigned long long)gap);
        run_search(P0, pref, xonly, gap, up, T, count, skip, max_n, have_k, kval,
                    custom?"search":"phase 2", &found);
    } else {
        foundlist f1; uint64_t l1[1]; f1.list=l1; f1.n=0; f1.cap=1;
        run_search(P0, pref, xonly, 1, up, T, 1, 0, 0, have_k, kval, "phase 1", &f1);
        if(f1.n==0){ fprintf(stderr,"phase 1 found nothing\n"); return 1; }
        gap = f1.list[0];
        printf("\n==> gap = %llu\n",(unsigned long long)gap);
        skip = 1; { const char *o=get_opt(argc,argv,"skip"); if(o) skip=strtoull(o,0,10); }
        run_search(P0, pref, xonly, gap, up, T, count, skip, max_n, have_k, kval, "phase 2", &found);
    }

    const char *side = up? "above" : "below";
    printf("\n================ RESULT ================\n");
    printf("start   = %s\n", h);
    printf("prefix  = %s%s\n", pref, xonly?"  (x only)":"");
    printf("gap     = %llu   direction = %s\n", (unsigned long long)gap, up?"UP":"DOWN");
    if(found.n==0) printf("no hit found\n");
    pt G; G.x=fe_from_hex("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    { u256 u=u256_from_hex("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"); memcpy(G.y.v,u.v,sizeof u.v); }
    G.inf=0;
    for(int i=0;i<found.n;i++){
        uint64_t n=found.list[i]; uint64_t d=n*gap;
        pt m = signed_mul(G, d, up?1:-1); pt t2 = aff_add(P0, m);
        char pub[67]; hexpub(t2,pub);
        printf("n = %llu   total distance %s start = %llu\n",(unsigned long long)n, side, (unsigned long long)d);
        printf("found   = %s\n", pub);
        if(have_k){ u256 sc = scalar_add_signed(kval, n, gap, up?1:-1); char sh[65]; u256_to_hex(sc,sh);
                    printf("scalar  = %s\n", sh); }
    }
    return 0;
}
