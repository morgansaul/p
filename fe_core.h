#ifndef FE_CORE_H
#define FE_CORE_H
#include <stdint.h>
#include <string.h>
#ifdef __CUDACC__
#define QUAL __host__ __device__ __forceinline__
#else
#define QUAL static inline
#endif

typedef unsigned __int128 u128;
typedef struct { uint64_t v[4]; } fe;          /* mod p, little-endian 64-bit limbs */

static const uint64_t C64 = 0x1000003D1ULL;    /* 2^256 mod p = 2^32 + 977 */
static const uint64_t P64[4] = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                                 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};

QUAL void cond_sub_p(fe *r){
    uint64_t t[4]; u128 b=0;
    for(int i=0;i<4;i++){ u128 cur=(u128)r->v[i]-P64[i]-b; t[i]=(uint64_t)cur; b=(cur>>127)&1; }
    if(!b) memcpy(r->v,t,sizeof t);
}
QUAL fe fe_add(fe a, fe b){
    fe r; u128 c=0;
    for(int i=0;i<4;i++){ u128 cur=(u128)a.v[i]+b.v[i]+c; r.v[i]=(uint64_t)cur; c=cur>>64; }
    if(c){ u128 cur=(u128)r.v[0]+C64; r.v[0]=(uint64_t)cur; u128 k=cur>>64;
           for(int i=1;i<4;i++){ cur=(u128)r.v[i]+k; r.v[i]=(uint64_t)cur; k=cur>>64; } }
    cond_sub_p(&r); return r;
}
QUAL fe fe_sub(fe a, fe b){
    fe r; u128 br=0;
    for(int i=0;i<4;i++){ u128 cur=(u128)a.v[i]-b.v[i]-br; r.v[i]=(uint64_t)cur; br=(cur>>127)&1; }
    if(br){ u128 c=0; for(int i=0;i<4;i++){ u128 cur=(u128)r.v[i]+P64[i]+c; r.v[i]=(uint64_t)cur; c=cur>>64; } }
    return r;
}
QUAL fe fe_mul(fe a, fe b){
    uint64_t t[8]; for(int i=0;i<8;i++) t[i]=0;
    for(int i=0;i<4;i++){
        u128 carry=0;
        for(int j=0;j<4;j++){
            u128 cur=(u128)a.v[i]*b.v[j]+t[i+j]+carry;
            t[i+j]=(uint64_t)cur; carry=cur>>64;
        }
        t[i+4]=(uint64_t)carry;
    }
    uint64_t p5[5]={0,0,0,0,0}; u128 carry=0;
    for(int i=0;i<4;i++){ u128 cur=(u128)t[4+i]*C64+carry; p5[i]=(uint64_t)cur; carry=cur>>64; }
    p5[4]=(uint64_t)carry;
    uint64_t r[4]; u128 c2=0;
    for(int i=0;i<4;i++){ u128 cur=(u128)t[i]+p5[i]+c2; r[i]=(uint64_t)cur; c2=cur>>64; }
    uint64_t r4 = p5[4] + (uint64_t)c2;
    u128 cur=(u128)r[0]+(u128)r4*C64; r[0]=(uint64_t)cur; u128 cc=cur>>64;
    for(int i=1;i<4;i++){ cur=(u128)r[i]+cc; r[i]=(uint64_t)cur; cc=cur>>64; }
    while(cc){ u128 c2b=cc; cc=0;
        u128 cur2=(u128)r[0]+c2b*C64; r[0]=(uint64_t)cur2; u128 k=cur2>>64;
        for(int i=1;i<4;i++){ cur2=(u128)r[i]+k; r[i]=(uint64_t)cur2; k=cur2>>64; }
        cc=k;
    }
    fe o; memcpy(o.v,r,sizeof r); cond_sub_p(&o); return o;
}
QUAL fe fe_sqr(fe a){ return fe_mul(a,a); }
QUAL fe fe_sqrn(fe a,int n){ for(int i=0;i<n;i++) a=fe_sqr(a); return a; }
QUAL fe fe_inv(fe a){
    fe x2=fe_mul(fe_sqr(a),a);
    fe x3=fe_mul(fe_sqr(x2),a);
    fe x6=fe_mul(fe_sqrn(x3,3),x3);
    fe x9=fe_mul(fe_sqrn(x6,3),x3);
    fe x11=fe_mul(fe_sqrn(x9,2),x2);
    fe x22=fe_mul(fe_sqrn(x11,11),x11);
    fe x44=fe_mul(fe_sqrn(x22,22),x22);
    fe x88=fe_mul(fe_sqrn(x44,44),x44);
    fe x176=fe_mul(fe_sqrn(x88,88),x88);
    fe x220=fe_mul(fe_sqrn(x176,44),x44);
    fe x223=fe_mul(fe_sqrn(x220,3),x3);
    fe t=fe_mul(fe_sqrn(x223,23),x22);
    t=fe_mul(fe_sqrn(t,5),a);
    t=fe_mul(fe_sqrn(t,3),x2);
    t=fe_mul(fe_sqrn(t,2),a);
    return t;
}
QUAL fe fe_one_v(void){ fe r={{1,0,0,0}}; return r; }
QUAL fe fe_zero_v(void){ fe r={{0,0,0,0}}; return r; }
QUAL int fe_is_zero(fe a){ return a.v[0]==0&&a.v[1]==0&&a.v[2]==0&&a.v[3]==0; }
QUAL int fe_eq(fe a, fe b){ return a.v[0]==b.v[0]&&a.v[1]==b.v[1]&&a.v[2]==b.v[2]&&a.v[3]==b.v[3]; }
QUAL fe fe_from_u64(uint64_t x){ fe r={{x,0,0,0}}; return r; }
QUAL fe fe_pow(fe a, const uint64_t *e){
    fe r=fe_one_v();
    for(int limb=3; limb>=0; limb--)
        for(int bit=63; bit>=0; bit--){
            r=fe_sqr(r);
            if((e[limb]>>bit)&1) r=fe_mul(r,a);
        }
    return r;
}
#endif
