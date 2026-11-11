/* ---------------------------------------------------------------------------
   Host-side big-integer / EC helpers built on the verified fe (mod p) type.
   Everything here is tested against the Python reference in p.py.
--------------------------------------------------------------------------- */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static const uint64_t N64[4] = {0xbfd25e8cd0364141ULL, 0xbaaedce6af48a03bULL,
                                 0xfffffffffffffffeULL, 0xffffffffffffffffULL};

/* ---- generic 256-bit integer (used only for scalar display, mod N) ---- */
typedef struct { uint64_t v[4]; } u256;

static int u256_cmp(u256 a, u256 b){
    for(int i=3;i>=0;i--){ if(a.v[i]!=b.v[i]) return a.v[i]>b.v[i]?1:-1; }
    return 0;
}
static u256 u256_sub(u256 a, u256 b){      /* a - b, a >= b assumed */
    u256 r; u128 br=0;
    for(int i=0;i<4;i++){ u128 cur=(u128)a.v[i]-b.v[i]-br; r.v[i]=(uint64_t)cur; br=(cur>>127)&1; }
    return r;
}
static u256 u256_add(u256 a, u256 b){
    u256 r; u128 c=0;
    for(int i=0;i<4;i++){ u128 cur=(u128)a.v[i]+b.v[i]+c; r.v[i]=(uint64_t)cur; c=cur>>64; }
    return r;
}
static u256 u256_from_u128(u128 x){ u256 r={{(uint64_t)x,(uint64_t)(x>>64),0,0}}; return r; }

/* k (mod N, k < N) plus/minus (n*gap) (both uint64_t, product fits in u128) */
static u256 scalar_add_signed(u256 k, uint64_t n, uint64_t gap, int sign /* +1 or -1 */){
    u128 prod = (u128)n * (u128)gap;
    u256 d = u256_from_u128(prod);
    u256 N = {{N64[0],N64[1],N64[2],N64[3]}};
    if(sign > 0){
        u256 r = u256_add(k, d);
        if(u256_cmp(r,N) >= 0) r = u256_sub(r,N);
        return r;
    } else {
        if(u256_cmp(k,d) >= 0) return u256_sub(k,d);
        return u256_sub(u256_add(k,N), d);
    }
}
static void u256_to_hex(u256 a, char *out /* 65 bytes */){
    for(int i=3;i>=0;i--) sprintf(out + (3-i)*16, "%016llx", (unsigned long long)a.v[i]);
}
static u256 u256_from_hex(const char *s){   /* exactly 64 hex chars */
    u256 r={{0,0,0,0}};
    for(int i=0;i<64;i++){ int c=s[i]; int d = (c<='9')?c-'0':((c|32)-'a'+10);
        int limb=(63-i)/16, shift=((63-i)%16)*4; r.v[limb] |= (uint64_t)d<<shift; }
    return r;
}

/* fe_pow, fe_one_v, fe_zero_v, fe_is_zero, fe_eq, fe_from_u64 come from fe_core.h */

/* ---- field element from/to hex (reuses fe = mod p) ---- */
static fe fe_from_hex(const char *s){ u256 u=u256_from_hex(s); fe r; memcpy(r.v,u.v,sizeof r.v); return r; }
static void fe_to_hex(fe a, char *out){ u256 u; memcpy(u.v,a.v,sizeof u.v); u256_to_hex(u,out); }

/* ---- affine EC point ops (mod p) ---- */
typedef struct { fe x,y; int inf; } pt;


static pt aff_add(pt P, pt Q){
    if(P.inf) return Q;
    if(Q.inf) return P;
    if(fe_eq(P.x,Q.x)){
        fe s=fe_add(P.y,Q.y);
        if(fe_is_zero(s)){ pt r; memset(&r,0,sizeof r); r.inf=1; return r; }
        fe num=fe_add(fe_add(fe_sqr(P.x),fe_sqr(P.x)),fe_sqr(P.x));     /* 3x^2 */
        fe den=fe_add(P.y,P.y);
        fe l=fe_mul(num, fe_inv(den));
        fe x3=fe_sub(fe_sub(fe_sqr(l),P.x),P.x);
        fe y3=fe_sub(fe_mul(l,fe_sub(P.x,x3)),P.y);
        pt r={x3,y3,0}; return r;
    }
    fe l=fe_mul(fe_sub(Q.y,P.y), fe_inv(fe_sub(Q.x,P.x)));
    fe x3=fe_sub(fe_sub(fe_sqr(l),P.x),Q.x);
    fe y3=fe_sub(fe_mul(l,fe_sub(P.x,x3)),P.y);
    pt r={x3,y3,0}; return r;
}

static const uint64_t G64x[4] = {0x59f2815b16f81798ULL,0x029bfcdb2dce28d9ULL,0x55a06295ce870b07ULL,0x79be667ef9dcbbacULL};
static const uint64_t G64y[4] = {0x9c47d08ffb10d4b8ULL,0xfd17b448a6855419ULL,0x5da4fbfc0e1108a8ULL,0x483ada7726a3c465ULL};

static pt mul_pt(pt G, uint64_t k_lo /* supports scalars up to 64 bits, enough for gap*T etc */){
    pt R; memset(&R,0,sizeof R); R.inf=1;
    pt A=G;
    for(int bit=0; bit<64; bit++){
        if((k_lo>>bit)&1) R=aff_add(R,A);
        A=aff_add(A,A);
    }
    return R;
}
static pt mul_pt256(pt G, const uint64_t *k /* 4 limbs */){
    pt R; memset(&R,0,sizeof R); R.inf=1;
    pt A=G;
    for(int limb=0; limb<4; limb++)
        for(int bit=0; bit<64; bit++){
            if((k[limb]>>bit)&1) R=aff_add(R,A);
            A=aff_add(A,A);
        }
    return R;
}

/* decompress a 33-byte compressed pubkey given as 66 hex chars */
static int decompress(const char *h, pt *out){
    if(strlen(h)!=66) return 0;
    int parity = (h[0]=='0'&&h[1]=='3') ? 1 : ((h[0]=='0'&&h[1]=='2')?0:-1);
    if(parity<0) return 0;
    fe x = fe_from_hex(h+2);
    fe rhs = fe_add(fe_mul(fe_mul(x,x),x), fe_from_u64(7));
    static const uint64_t PPOW[4] = {0xffffffffbfffff0cULL,0xffffffffffffffffULL,0xffffffffffffffffULL,0x3fffffffffffffffULL}; /* (p+1)/4 */
    fe y = fe_pow(rhs, PPOW);
    if(((int)(y.v[0]&1)) != parity){ y = fe_sub(fe_from_u64(0), y); }
    /* verify */
    fe lhs=fe_sqr(y);
    if(!fe_eq(lhs,rhs)) return 0;
    out->x=x; out->y=y; out->inf=0;
    return 1;
}
static void hexpub(pt P, char *out /* 67 bytes */){
    char xh[65]; fe_to_hex(P.x, xh);
    sprintf(out, "%02x%s", 2 | (int)(P.y.v[0]&1), xh);
}

/* ---- Jacobian batch stepping (B successive points P0+step, P0+2*step, ...) with ONE
   shared inversion, matching block() in p.py (handles doubling / infinity exactly). ---- */
typedef struct { fe X,Y,Z; int inf; } jpt;

static jpt jpt_madd(jpt P, fe x2, fe y2){     /* P (Jacobian) + affine (x2,y2), P != +-affine(x2,y2) */
    fe Z1Z1=fe_sqr(P.Z);
    fe U2=fe_mul(x2,Z1Z1);
    fe S2=fe_mul(fe_mul(y2,P.Z),Z1Z1);
    fe H=fe_sub(U2,P.X);
    fe HH=fe_sqr(H);
    fe I=fe_add(HH,HH); I=fe_add(I,I);
    fe J=fe_mul(H,I);
    fe r=fe_sub(S2,P.Y); r=fe_add(r,r);
    fe V=fe_mul(P.X,I);
    fe X3=fe_sub(fe_sub(fe_sqr(r),J),fe_add(V,V));
    fe YJ=fe_mul(P.Y,J);
    fe Y3=fe_sub(fe_mul(r,fe_sub(V,X3)),fe_add(YJ,YJ));
    fe Z3=fe_sub(fe_sub(fe_sqr(fe_add(P.Z,H)),Z1Z1),HH);
    jpt R={X3,Y3,Z3,0}; return R;
}

/* B successive affine points starting at P0+step; returns malloc'd array of B pts (inf marked) */
static pt *block_batch(pt P0, pt step, int B){
    jpt *j = (jpt*)malloc(B*sizeof(jpt));
    jpt cur; cur.X=P0.x; cur.Y=P0.y; cur.Z=fe_one_v(); cur.inf=P0.inf;
    for(int i=0;i<B;i++){
        if(cur.inf){ jpt r; r.X=step.x; r.Y=step.y; r.Z=fe_one_v(); r.inf=step.inf; j[i]=cur=r; continue; }
        fe Z1Z1=fe_sqr(cur.Z);
        fe H=fe_sub(fe_mul(step.x,Z1Z1), cur.X);
        if(fe_is_zero(H)){
            /* doubling or P+(-P): fall back to exact affine add */
            pt aff; fe zi=fe_inv(cur.Z); fe zi2=fe_sqr(zi);
            aff.x=fe_mul(cur.X,zi2); aff.y=fe_mul(fe_mul(cur.Y,zi2),zi); aff.inf=0;
            pt nxt = aff_add(aff, step);
            jpt r; if(nxt.inf){ r.inf=1; } else { r.X=nxt.x; r.Y=nxt.y; r.Z=fe_one_v(); r.inf=0; }
            j[i]=cur=r; continue;
        }
        cur = jpt_madd(cur, step.x, step.y);
        j[i]=cur;
    }
    /* batch invert all non-infinity Z's */
    fe *acc = (fe*)malloc(B*sizeof(fe)); fe a=fe_one_v();
    for(int i=0;i<B;i++){ acc[i]=a; if(!j[i].inf) a=fe_mul(a,j[i].Z); }
    fe inv=fe_inv(a);
    pt *out = (pt*)malloc(B*sizeof(pt));
    for(int i=B-1;i>=0;i--){
        if(j[i].inf){ memset(&out[i],0,sizeof(pt)); out[i].inf=1; continue; }
        fe zi=fe_mul(inv,acc[i]); inv=fe_mul(inv,j[i].Z);
        fe zi2=fe_sqr(zi);
        out[i].x=fe_mul(j[i].X,zi2);
        out[i].y=fe_mul(fe_mul(j[i].Y,zi2),zi);
        out[i].inf=0;
    }
    free(j); free(acc);
    return out;
}

/* detect if P0 = a * S for a tiny signed integer |a| <= c_max (S assumed = gap*G) */
static int64_t small_multiple(pt P0, pt S, int64_t c_max){
    if(fe_eq(P0.x,S.x)) return fe_eq(P0.y,S.y) ? 1 : -1;
    int64_t done=0; pt cur=S;
    while(done < c_max){
        int64_t m = (c_max-done) < 65536 ? (c_max-done) : 65536;
        pt *pts = block_batch(cur, S, (int)m);
        for(int64_t i=0;i<m;i++){
            if(!pts[i].inf && fe_eq(pts[i].x, P0.x)){
                int64_t c = done + i + 2;
                int64_t sgn = fe_eq(pts[i].y, P0.y) ? 1 : -1;
                free(pts);
                return sgn*c;
            }
        }
        cur = pts[m-1];
        free(pts);
        done += m;
    }
    return 0; /* sentinel: not found / treat 0 as "none" since a==0 impossible (P0 != infinity in practice) */
}
