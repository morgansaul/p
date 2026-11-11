/* ---------------------------------------------------------------------------
   Shared per-group search logic. This file is included BOTH by the CPU
   fallback path (plain C loop, usable immediately and used here to test
   correctness end-to-end) and, almost unchanged, by the CUDA __global__
   kernel (GROUP_KERNEL.cu), so a correctness fix in one effectively fixes
   both. HALF is a compile-time constant (recompile with -DHALF=N to change
   the group size / inversion-sharing factor).
--------------------------------------------------------------------------- */
#ifndef HALF
#define HALF 128
#endif
#define GS (2*HALF+1)

/* mask/tgt: mask[0..7]/tgt[0..7] match x (256 bits), mask[8]/tgt[8] (0 or
   0x02/0x03 pattern) optionally also match the parity byte. */
typedef struct {
    uint32_t mask[9], tgt[9];
} prefmatch;

static void pref_to_mask(const char *prefix, int xonly, prefmatch *pm){
    int n = (int)strlen(prefix);
    int shift = (xonly ? 256 : 264) - 4*n;
    /* build as a 264-bit big number (9 x 32-bit words, little endian) then shift */
    u128 lo=0; uint32_t top=0;
    /* value = int(prefix,16) << shift; representable in <=264 bits since n<=66 */
    /* do it with a small bignum: up to 9 32-bit words */
    uint32_t val[9]={0,0,0,0,0,0,0,0,0};
    /* parse prefix as hex into a 264-bit number first (right-aligned), then shift left */
    uint32_t num[9]={0,0,0,0,0,0,0,0,0};
    for(int i=0;i<n;i++){
        int c=prefix[i]; int d=(c<='9')?c-'0':((c|32)-'a'+10);
        /* num = num*16 + d, treating num as 9x32 little-endian */
        uint64_t carry=d;
        for(int w=0; w<9 && carry; w++){
            uint64_t cur=(uint64_t)num[w]*16 + carry;
            num[w]=(uint32_t)cur; carry=cur>>32;
        }
    }
    /* shift num left by `shift` bits into val (both as 9x32 little-endian, bit 0 = LSB of word0) */
    int word_shift = shift/32, bit_shift = shift%32;
    for(int w=8; w>=0; w--){
        int src = w - word_shift;
        uint32_t lo_ = (src>=0 && src<9) ? num[src] : 0;
        uint32_t hi_ = (bit_shift && src-1>=0 && src-1<9) ? num[src-1] : 0;
        val[w] = bit_shift ? ((lo_<<bit_shift) | (hi_>>(32-bit_shift))) : lo_;
    }
    uint32_t mask_full[9]={0,0,0,0,0,0,0,0,0};
    /* mask = ((1<<(4*n))-1) << shift, same shifting procedure applied to a run of 1 bits */
    int bits = 4*n;
    uint32_t onesrc[9]={0,0,0,0,0,0,0,0,0};
    for(int b=0;b<bits && b<288;b++) onesrc[b/32] |= (1u<<(b%32));
    for(int w=8; w>=0; w--){
        int src = w - word_shift;
        uint32_t lo_ = (src>=0 && src<9) ? onesrc[src] : 0;
        uint32_t hi_ = (bit_shift && src-1>=0 && src-1<9) ? onesrc[src-1] : 0;
        mask_full[w] = bit_shift ? ((lo_<<bit_shift) | (hi_>>(32-bit_shift))) : lo_;
    }
    for(int i=0;i<9;i++){ pm->mask[i]=mask_full[i]; pm->tgt[i]=val[i]; }
    (void)lo; (void)top;
}

/* x's 256 bits packed as 8x32 words (little endian) from the fe's 4x64 limbs,
   plus a 9th word whose low bits hold the parity byte (0x02|y&1). */
static void pt_to_words(fe x, int parity, uint32_t *w9){
    for(int i=0;i<4;i++){ w9[2*i]=(uint32_t)x.v[i]; w9[2*i+1]=(uint32_t)(x.v[i]>>32); }
    w9[8]=(uint32_t)parity;
}
static int prefix_match(const uint32_t *w9, const prefmatch *pm){
    for(int i=0;i<8;i++) if((w9[i]&pm->mask[i])!=pm->tgt[i]) return 0;
    if(pm->mask[8]) return (w9[8]&pm->mask[8])==pm->tgt[8];
    return 1;
}

/* One group's work: centre point C (affine), a precomputed table tab[1..HALF]
   of i*gap*G (affine, i=1..HALF), the fixed prefix test, and a callback-like
   output array. Returns the number of hits found in this group (<=2*HALF+1),
   writing their multiplier n (absolute, 1-based, as per run loop's numbering)
   into hits[]. g is this group's index (0-based); its multipliers span
   g*GS+1 .. g*GS+GS. up selects direction (true: n*gap*G is added going
   "forward" in n meaning distances above start; CHECK macro same as kernel). */
static int process_group(fe cx, fe cy, const fe *tabx, const fe *taby,
                          const prefmatch *pm, uint64_t g, int up,
                          uint64_t *hits_out, int max_hits)
{
    int nh = 0;
    uint64_t nc = g*(uint64_t)GS + HALF + 1;
    uint32_t w9[9];

    /* centre point */
    pt_to_words(cx, 2 | (int)(cy.v[0]&1), w9);
    if(prefix_match(w9,pm) && nh<max_hits) hits_out[nh++] = nc;

    fe acc = fe_one_v();
    fe pre[HALF];                          /* fixed-size: no malloc, device-safe */
    for(int i=0;i<HALF;i++){
        fe dx = fe_sub(tabx[i], cx);
        if(fe_is_zero(dx)) dx = fe_one_v();
        pre[i] = acc;
        acc = fe_mul(acc, dx);
    }
    fe inv = fe_inv(acc);
    for(int i=HALF-1;i>=0;i--){
        fe dx = fe_sub(tabx[i], cx);
        if(fe_is_zero(dx)) dx = fe_one_v();
        fe idx = fe_mul(inv, pre[i]);
        inv = fe_mul(inv, dx);
        uint64_t m = i+1;

        /* C + (i+1)*step */
        fe lp = fe_mul(fe_sub(taby[i], cy), idx);
        fe xp = fe_sub(fe_sub(fe_sqr(lp), cx), tabx[i]);
        fe yp = fe_sub(fe_mul(lp, fe_sub(cx, xp)), cy);
        pt_to_words(xp, 2 | (int)(yp.v[0]&1), w9);
        if(prefix_match(w9,pm) && nh<max_hits) hits_out[nh++] = up ? nc+m : nc-m;

        /* C - (i+1)*step */
        fe lm = fe_mul(fe_sub(fe_zero_v(), fe_add(taby[i], cy)), idx);
        fe xm = fe_sub(fe_sub(fe_sqr(lm), cx), tabx[i]);
        fe ym = fe_sub(fe_mul(lm, fe_sub(cx, xm)), cy);
        pt_to_words(xm, 2 | (int)(ym.v[0]&1), w9);
        if(prefix_match(w9,pm) && nh<max_hits) hits_out[nh++] = up ? nc-m : nc+m;
    }
    return nh;
}
