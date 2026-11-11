/* ---------------------------------------------------------------------------
   The actual search kernel body, shared between:
   - the CUDA __global__ kernel (one call = one GPU thread), and
   - the CPU/OpenMP fallback (one call = one loop iteration),
   via the KSEARCH macro below, so there is exactly one place where the
   per-thread "iters" loop and centre-advance logic live. Both paths call
   process_group() from group_logic.h for the per-group work, already
   verified against an independent brute-force scan.
--------------------------------------------------------------------------- */
#define KSEARCH_BODY(cx,cy,tabx,taby,bigx,bigy,pm,Tval,iter_base,iters,up_,hit_count,hit_dist,max_hits) \
do { \
    fe _cx=(cx), _cy=(cy); \
    for(uint32_t _it=0; _it<(iters); _it++){ \
        uint64_t _g = ((uint64_t)(iter_base)+_it)*(uint64_t)(Tval) + (uint64_t)tid; \
        uint64_t _hits[2*HALF+1]; \
        int _nh = process_group(_cx,_cy,(tabx),(taby),(pm),_g,(up_),_hits, (int)(2*HALF+1)); \
        for(int _k=0;_k<_nh;_k++){ \
            uint32_t _idx = ATOMIC_INC(hit_count); \
            if(_idx < (max_hits)) (hit_dist)[_idx] = _hits[_k]; \
        } \
        fe _dx = fe_sub((bigx), _cx); if(fe_is_zero(_dx)) _dx = fe_one_v(); \
        fe _l = fe_mul(fe_sub((bigy), _cy), fe_inv(_dx)); \
        fe _nx = fe_sub(fe_sub(fe_sqr(_l), _cx), (bigx)); \
        fe _ny = fe_sub(fe_mul(_l, fe_sub(_cx, _nx)), _cy); \
        _cx=_nx; _cy=_ny; \
    } \
    (cx)=_cx; (cy)=_cy; \
} while(0)
