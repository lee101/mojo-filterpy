from max.gpu import global_idx
from std.sys import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


@export("probe_width")
def probe_width() abi("C") -> Int:
    return W


@export("probe_add_scaled_repeat")
def probe_add_scaled_repeat(
    dst_addr: Int, src_addr: Int, scale: Float64, count: Int, iters: Int
) abi("C") -> Int:
    var dst = p(dst_addr)
    var src = p(src_addr)
    for _ in range(iters):
        var i = 0
        var scales = SIMD[DType.float64, W](scale)
        while i + W <= count:
            dst.store(
                i, dst.load[width=W](i) + scales * src.load[width=W](i)
            )
            i += W
        while i < count:
            dst[i] += scale * src[i]
            i += 1
    return 0


@export("probe_loop_only")
def probe_loop_only(dst_addr: Int, count: Int, iters: Int) abi("C") -> Int:
    var dst = p(dst_addr)
    var acc = 0.0
    for _ in range(iters):
        var i = 0
        while i + W <= count:
            acc += dst.load[width=W](i).reduce_add()
            i += W
        while i < count:
            acc += dst[i]
            i += 1
    return Int(acc)


@export("probe_call_overhead")
def probe_call_overhead(dst_addr: Int, iters: Int) abi("C") -> Int:
    var dst = p(dst_addr)
    for _ in range(iters):
        dst[0] = dst[0] + 1.0
    return 0


@export("probe_gemm_reg")
def probe_gemm_reg(
    c_addr: Int, a_addr: Int, b_addr: Int, d_addr: Int, rows: Int, inner: Int, cols: Int
) abi("C"):
    var c = p(c_addr)
    var a = p(a_addr)
    var b = p(b_addr)
    var d = p(d_addr)
    for i in range(rows):
        var ar = a + i * inner
        var cr = c + i * cols
        var dr = d + i * cols
        var j = 0
        while j + W <= cols:
            var acc = SIMD[DType.float64, W](0.0)
            for t in range(inner):
                acc += ar[t] * b.load[width=W](t * cols + j)
            cr.store(j, acc + dr.load[width=W](j))
            j += W
        while j < cols:
            var s = 0.0
            for t in range(inner):
                s += ar[t] * b[t * cols + j]
            cr[j] = s + dr[j]
            j += 1
