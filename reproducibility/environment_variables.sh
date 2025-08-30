#!/bin/bash

# 1) Force single-threaded math everywhere (most important for stability)
export OMP_NUM_THREADS=1            # OpenMP runtime threads :contentReference[oaicite:0]{index=0}
export OPENBLAS_NUM_THREADS=1       # OpenBLAS threads :contentReference[oaicite:1]{index=1}
export MKL_NUM_THREADS=1            # Intel oneMKL threads
export BLIS_NUM_THREADS=1           # BLIS threads
export VECLIB_MAXIMUM_THREADS=1     # Apple Accelerate threads (macOS) :contentReference[oaicite:2]{index=2}
export NUMEXPR_NUM_THREADS=1        # numexpr threads (if installed) :contentReference[oaicite:3]{index=3}
export OMP_DYNAMIC=false            # don’t change threads at runtime

# 2) Make MKL pick a stable code path (helps cross-CPU reproducibility)
export MKL_CBWR=COMPATIBLE          # avoids CPU-dependent approx instructions; SSE2 path :contentReference[oaicite:4]{index=4}

# 3) OpenMP reductions: deterministic order (Intel runtimes)
export KMP_DETERMINISTIC_REDUCTION=true    # stable floating-point reductions with fixed threads :contentReference[oaicite:5]{index=5}

# 4) Optional MKL knobs
export MKL_DISABLE_FAST_MM=1        # disable MKL’s fast mem mgr (sometimes helps stability; can slow small ops) :contentReference[oaicite:6]{index=6}
# export MKL_VERBOSE=1              # debug: prints which MKL kernels get used
