"""Dense filtering kernels exposed through one C ABI compilation unit."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.sys import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def dot(a: Ptr, b: Ptr, count: Int) -> Float64:
    var vectors = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= count:
        vectors += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var acc = vectors.reduce_add()
    while i < count:
        acc += a[i] * b[i]
        i += 1
    return acc


def copy_values(dst: Ptr, src: Ptr, count: Int):
    var i = 0
    while i + W <= count:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < count:
        dst[i] = src[i]
        i += 1


def fill_values(dst: Ptr, value: Float64, count: Int):
    var i = 0
    var values = SIMD[DType.float64, W](value)
    while i + W <= count:
        dst.store(i, values)
        i += W
    while i < count:
        dst[i] = value
        i += 1


def add_scaled(dst: Ptr, src: Ptr, scale: Float64, count: Int):
    var i = 0
    var scales = SIMD[DType.float64, W](scale)
    while i + W <= count:
        dst.store(i, dst.load[width=W](i) + scales * src.load[width=W](i))
        i += W
    while i < count:
        dst[i] += scale * src[i]
        i += 1


def row_gemm(
    dst: Ptr, coeffs: Ptr, src: Ptr, inner: Int, cols: Int, scale: Float64
):
    var j = 0
    var scales = SIMD[DType.float64, W](scale)
    while j + W <= cols:
        var acc = SIMD[DType.float64, W](0.0)
        for t in range(inner):
            acc += coeffs[t] * src.load[width=W](t * cols + j)
        dst.store(j, scales * acc)
        j += W
    while j < cols:
        var accs = 0.0
        for t in range(inner):
            accs += coeffs[t] * src[t * cols + j]
        dst[j] = scale * accs
        j += 1


def row_gemm_bias(
    dst: Ptr, coeffs: Ptr, src: Ptr, bias: Ptr, inner: Int, cols: Int, scale: Float64
):
    var j = 0
    var scales = SIMD[DType.float64, W](scale)
    while j + W <= cols:
        var acc = SIMD[DType.float64, W](0.0)
        for t in range(inner):
            acc += coeffs[t] * src.load[width=W](t * cols + j)
        dst.store(j, scales * acc + bias.load[width=W](j))
        j += W
    while j < cols:
        var accs = 0.0
        for t in range(inner):
            accs += coeffs[t] * src[t * cols + j]
        dst[j] = scale * accs + bias[j]
        j += 1


def transpose(src: Ptr, dst: Ptr, rows: Int, cols: Int):
    var i = 0
    while i < rows:
        var s = src + i * cols
        var j = 0
        while j < cols:
            dst[j * rows + i] = s[j]
            j += 1
        i += 1


def invert(a: Ptr, inv: Ptr, work: Ptr, n: Int) -> Bool:
    copy_values(work, a, n * n)
    fill_values(inv, 0.0, n * n)
    for i in range(n):
        inv[i * n + i] = 1.0
    for col in range(n):
        var pivot = col
        var largest = abs(work[col * n + col])
        for row in range(col + 1, n):
            var candidate = abs(work[row * n + col])
            if candidate > largest:
                largest = candidate
                pivot = row
        if largest <= 1.0e-15:
            return False
        if pivot != col:
            var j = 0
            while j + W <= n:
                var t = work.load[width=W](col * n + j)
                work.store(col * n + j, work.load[width=W](pivot * n + j))
                work.store(pivot * n + j, t)
                t = inv.load[width=W](col * n + j)
                inv.store(col * n + j, inv.load[width=W](pivot * n + j))
                inv.store(pivot * n + j, t)
                j += W
            while j < n:
                var scalar = work[col * n + j]
                work[col * n + j] = work[pivot * n + j]
                work[pivot * n + j] = scalar
                scalar = inv[col * n + j]
                inv[col * n + j] = inv[pivot * n + j]
                inv[pivot * n + j] = scalar
                j += 1
        var scale = work[col * n + col]
        var j = 0
        var scales = SIMD[DType.float64, W](scale)
        while j + W <= n:
            work.store(
                col * n + j, work.load[width=W](col * n + j) / scales
            )
            inv.store(col * n + j, inv.load[width=W](col * n + j) / scales)
            j += W
        while j < n:
            work[col * n + j] /= scale
            inv[col * n + j] /= scale
            j += 1
        for row in range(n):
            if row == col:
                continue
            var factor = work[row * n + col]
            if factor != 0.0:
                j = 0
                var factors = SIMD[DType.float64, W](factor)
                while j + W <= n:
                    work.store(
                        row * n + j,
                        work.load[width=W](row * n + j)
                        - factors * work.load[width=W](col * n + j),
                    )
                    inv.store(
                        row * n + j,
                        inv.load[width=W](row * n + j)
                        - factors * inv.load[width=W](col * n + j),
                    )
                    j += W
                while j < n:
                    work[row * n + j] -= factor * work[col * n + j]
                    inv[row * n + j] -= factor * inv[col * n + j]
                    j += 1
    return True


@export("mfp_predict")
def mfp_predict(
    x_addr: Int,
    p_addr: Int,
    f_addr: Int,
    q_addr: Int,
    b_addr: Int,
    u_addr: Int,
    x_work_addr: Int,
    p_work_addr: Int,
    n: Int,
    nu: Int,
    alpha_sq: Float64,
) abi("C"):
    var x = p(x_addr)
    var cov = p(p_addr)
    var f = p(f_addr)
    var q = p(q_addr)
    var xw = p(x_work_addr)
    var work = p(p_work_addr)
    var pw = work
    var ft = work + n * n
    for i in range(n):
        xw[i] = dot(f + i * n, x, n)
    if nu > 0:
        var b = p(b_addr)
        var u = p(u_addr)
        for i in range(n):
            xw[i] += dot(b + i * nu, u, nu)
    for i in range(n):
        row_gemm(pw + i * n, f + i * n, cov, n, n, 1.0)
    transpose(f, ft, n, n)
    for i in range(n):
        row_gemm_bias(cov + i * n, pw + i * n, ft, q + i * n, n, n, alpha_sq)
    copy_values(x, xw, n)


@export("mfp_linear_update")
def mfp_linear_update(
    x_addr: Int,
    p_addr: Int,
    h_addr: Int,
    r_addr: Int,
    y_addr: Int,
    k_addr: Int,
    s_addr: Int,
    si_addr: Int,
    work_addr: Int,
    n: Int,
    m: Int,
) abi("C") -> Int:
    var x = p(x_addr)
    var cov = p(p_addr)
    var h = p(h_addr)
    var r = p(r_addr)
    var y = p(y_addr)
    var gain = p(k_addr)
    var s = p(s_addr)
    var si = p(si_addr)
    var work = p(work_addr)
    for i in range(n):
        for j in range(m):
            gain[i * m + j] = dot(cov + i * n, h + j * n, n)
    for i in range(m):
        row_gemm_bias(s + i * m, h + i * n, gain, r + i * m, n, m, 1.0)
    if not invert(s, si, work + 2 * n * n, m):
        return 0
    var product = work + 2 * n * n + m * m
    for i in range(n):
        row_gemm(product + i * m, gain + i * m, si, m, m, 1.0)
    copy_values(gain, product, n * m)
    for i in range(n):
        x[i] += dot(gain + i * m, y, m)
    var ikh = work
    var ap = work + n * n
    for i in range(n):
        row_gemm(ikh + i * n, gain + i * m, h, m, n, -1.0)
        ikh[i * n + i] += 1.0
    for i in range(n):
        row_gemm(ap + i * n, ikh + i * n, cov, n, n, 1.0)
    for i in range(n):
        row_gemm(product + i * m, gain + i * m, r, m, m, 1.0)
    for i in range(n):
        for j in range(n):
            cov[i * n + j] = (
                dot(ap + i * n, ikh + j * n, n)
                + dot(product + i * m, gain + j * m, m)
            )
    return 1


@export("mfp_unscented_transform")
def mfp_unscented_transform(
    sigmas_addr: Int,
    wm_addr: Int,
    wc_addr: Int,
    noise_addr: Int,
    mean_addr: Int,
    cov_addr: Int,
    dev_addr: Int,
    k: Int,
    n: Int,
) abi("C"):
    var sigmas = p(sigmas_addr)
    var wm = p(wm_addr)
    var wc = p(wc_addr)
    var noise = p(noise_addr)
    var mean = p(mean_addr)
    var cov = p(cov_addr)
    var dev = p(dev_addr)
    fill_values(mean, 0.0, n)
    for s in range(k):
        add_scaled(mean, sigmas + s * n, wm[s], n)
    for s in range(k):
        var sr = sigmas + s * n
        var dr = dev + s * n
        var j = 0
        while j + W <= n:
            dr.store(j, sr.load[width=W](j) - mean.load[width=W](j))
            j += W
        while j < n:
            dr[j] = sr[j] - mean[j]
            j += 1
    comptime BLOCK = 4 * W
    for i in range(n):
        var cr = cov + i * n
        var nr = noise + i * n
        var j = 0
        while j + BLOCK <= n:
            var a0 = nr.load[width=W](j)
            var a1 = nr.load[width=W](j + W)
            var a2 = nr.load[width=W](j + 2 * W)
            var a3 = nr.load[width=W](j + 3 * W)
            for s in range(k):
                var w = wc[s] * dev[s * n + i]
                var row = dev + s * n
                a0 += w * row.load[width=W](j)
                a1 += w * row.load[width=W](j + W)
                a2 += w * row.load[width=W](j + 2 * W)
                a3 += w * row.load[width=W](j + 3 * W)
            cr.store(j, a0)
            cr.store(j + W, a1)
            cr.store(j + 2 * W, a2)
            cr.store(j + 3 * W, a3)
            j += BLOCK
        while j + W <= n:
            var acc = nr.load[width=W](j)
            for s in range(k):
                acc += (wc[s] * dev[s * n + i]) * dev.load[width=W](s * n + j)
            cr.store(j, acc)
            j += W
        while j < n:
            var accs = nr[j]
            for s in range(k):
                accs += wc[s] * dev[s * n + i] * dev[s * n + j]
            cr[j] = accs
            j += 1


def ut_mean_gpu(
    sigmas: Ptr, wm: Ptr, mean: Ptr, k: Int32, n: Int32
):
    var j = Int(global_idx.x)
    var kk = Int(k)
    var nn = Int(n)
    if j < nn:
        var acc = 0.0
        for i in range(kk):
            acc += wm[i] * sigmas[i * nn + j]
        mean[j] = acc


def ut_cov_gpu(
    sigmas: Ptr,
    wc: Ptr,
    noise: Ptr,
    mean: Ptr,
    cov: Ptr,
    k: Int32,
    n: Int32,
):
    var index = Int(global_idx.x)
    var kk = Int(k)
    var nn = Int(n)
    if index < nn * nn:
        var row = index // nn
        var col = index - row * nn
        var acc = noise[index]
        for sidx in range(kk):
            acc += wc[sidx] * (sigmas[sidx * nn + row] - mean[row]) * (
                sigmas[sidx * nn + col] - mean[col]
            )
        cov[index] = acc


@export("mfp_unscented_transform_gpu")
def mfp_unscented_transform_gpu(
    sigmas_addr: Int,
    wm_addr: Int,
    wc_addr: Int,
    noise_addr: Int,
    mean_addr: Int,
    cov_addr: Int,
    dev_addr: Int,
    k: Int,
    n: Int,
) abi("C") -> Int:
    var sigmas = p(sigmas_addr)
    var wm = p(wm_addr)
    var wc = p(wc_addr)
    var noise = p(noise_addr)
    var mean = p(mean_addr)
    var cov = p(cov_addr)
    var dev = p(dev_addr)
    try:
        var ctx = DeviceContext()
        if ctx.api() == "cpu":
            mfp_unscented_transform(
                sigmas_addr,
                wm_addr,
                wc_addr,
                noise_addr,
                mean_addr,
                cov_addr,
                dev_addr,
                k,
                n,
            )
            return 0
        var d_sigmas = ctx.enqueue_create_buffer[DType.float64](k * n)
        var d_wm = ctx.enqueue_create_buffer[DType.float64](k)
        var d_wc = ctx.enqueue_create_buffer[DType.float64](k)
        var d_noise = ctx.enqueue_create_buffer[DType.float64](n * n)
        var d_mean = ctx.enqueue_create_buffer[DType.float64](n)
        var d_cov = ctx.enqueue_create_buffer[DType.float64](n * n)
        ctx.enqueue_copy(d_sigmas, sigmas)
        ctx.enqueue_copy(d_wm, wm)
        ctx.enqueue_copy(d_wc, wc)
        ctx.enqueue_copy(d_noise, noise)
        comptime block_size = 256
        ctx.enqueue_function[ut_mean_gpu](
            d_sigmas,
            d_wm,
            d_mean,
            Int32(k),
            Int32(n),
            grid_dim=(n + block_size - 1) // block_size,
            block_dim=block_size,
        )
        ctx.enqueue_function[ut_cov_gpu](
            d_sigmas,
            d_wc,
            d_noise,
            d_mean,
            d_cov,
            Int32(k),
            Int32(n),
            grid_dim=(n * n + block_size - 1) // block_size,
            block_dim=block_size,
        )
        ctx.enqueue_copy(mean, d_mean)
        ctx.enqueue_copy(cov, d_cov)
        ctx.synchronize()
        return 1
    except:
        return -1


@export("mfp_cross_variance")
def mfp_cross_variance(
    sx_addr: Int,
    x_addr: Int,
    sz_addr: Int,
    z_addr: Int,
    wc_addr: Int,
    cross_addr: Int,
    k: Int,
    nx: Int,
    nz: Int,
) abi("C"):
    var sx = p(sx_addr)
    var x = p(x_addr)
    var sz = p(sz_addr)
    var z = p(z_addr)
    var wc = p(wc_addr)
    var cross = p(cross_addr)
    for i in range(nx):
        for j in range(nz):
            var acc = 0.0
            for sidx in range(k):
                acc += wc[sidx] * (sx[sidx * nx + i] - x[i]) * (
                    sz[sidx * nz + j] - z[j]
                )
            cross[i * nz + j] = acc


@export("mfp_ukf_update")
def mfp_ukf_update(
    x_addr: Int,
    p_addr: Int,
    cross_addr: Int,
    s_addr: Int,
    y_addr: Int,
    k_addr: Int,
    si_addr: Int,
    work_addr: Int,
    nx: Int,
    nz: Int,
) abi("C") -> Int:
    var x = p(x_addr)
    var cov = p(p_addr)
    var cross = p(cross_addr)
    var s = p(s_addr)
    var y = p(y_addr)
    var gain = p(k_addr)
    var si = p(si_addr)
    var work = p(work_addr)
    if not invert(s, si, work, nz):
        return 0
    for i in range(nx):
        for j in range(nz):
            var acc = 0.0
            for a in range(nz):
                acc += cross[i * nz + a] * si[a * nz + j]
            gain[i * nz + j] = acc
    for i in range(nx):
        for j in range(nz):
            x[i] += gain[i * nz + j] * y[j]
    for i in range(nx):
        for j in range(nx):
            var acc = 0.0
            for a in range(nz):
                for bcol in range(nz):
                    acc += gain[i * nz + a] * s[a * nz + bcol] * gain[j * nz + bcol]
            cov[i * nx + j] -= acc
    return 1


@export("mfp_resample_sorted")
def mfp_resample_sorted(
    weights_addr: Int, positions_addr: Int, indexes_addr: Int, n: Int
) abi("C"):
    var weights = p(weights_addr)
    var positions = p(positions_addr)
    var indexes = ip(indexes_addr)
    var i = 0
    var j = 0
    var cumulative = weights[0]
    while i < n:
        if positions[i] < cumulative or j == n - 1:
            indexes[i] = Int64(j)
            i += 1
        else:
            j += 1
            cumulative += weights[j]


@export("mfp_resample_binary")
def mfp_resample_binary(
    cumulative_addr: Int,
    positions_addr: Int,
    indexes_addr: Int,
    table_addr: Int,
    n: Int,
    buckets: Int,
) abi("C"):
    var cumulative = p(cumulative_addr)
    var positions = p(positions_addr)
    var indexes = ip(indexes_addr)
    var table = ip(table_addr)
    var scale = Float64(buckets)

    # table[b] counts the entries whose bucket index is below b. Bucket index
    # is a monotone function of the value, so an entry at or below a position
    # always lands in a bucket at or below the position's bucket: the answer
    # is provably inside [table[b], table[b + 1]], which turns an
    # O(log n) search over the whole array into O(log(n / buckets)).
    var idx = 0
    for b in range(buckets + 2):
        while idx < n:
            var slot = Int(cumulative[idx] * scale)
            if slot < 0:
                slot = 0
            elif slot > buckets:
                slot = buckets
            if slot < b:
                idx += 1
            else:
                break
        table[b] = Int64(idx)
    for i in range(n):
        var position = positions[i]
        var b = Int(position * scale)
        if b < 0:
            b = 0
        elif b > buckets:
            b = buckets
        var lo = Int(table[b])
        var hi = Int(table[b + 1])
        while lo < hi:
            var mid = (lo + hi) // 2
            if position < cumulative[mid]:
                hi = mid
            else:
                lo = mid + 1
        indexes[i] = Int64(min(lo, n - 1))


@export("mfp_neff")
def mfp_neff(weights_addr: Int, n: Int) abi("C") -> Float64:
    var weights = p(weights_addr)
    var acc = 0.0
    for i in range(n):
        acc += weights[i] * weights[i]
    return 1.0 / acc
