"""Dense filtering kernels exposed through one C ABI compilation unit."""

from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext
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
    var pw = p(p_work_addr)
    for i in range(n):
        xw[i] = dot(f + i * n, x, n)
    if nu > 0:
        var b = p(b_addr)
        var u = p(u_addr)
        for i in range(n):
            xw[i] += dot(b + i * nu, u, nu)
    for i in range(n):
        fill_values(pw + i * n, 0.0, n)
        for a in range(n):
            add_scaled(pw + i * n, cov + a * n, f[i * n + a], n)
    for i in range(n):
        for j in range(n):
            cov[i * n + j] = (
                alpha_sq * dot(pw + i * n, f + j * n, n) + q[i * n + j]
            )
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
        copy_values(s + i * m, r + i * m, m)
        for a in range(n):
            add_scaled(s + i * m, gain + a * m, h[i * n + a], m)
    if not invert(s, si, work + 2 * n * n, m):
        return 0
    var product = work + 2 * n * n + m * m
    for i in range(n):
        fill_values(product + i * m, 0.0, m)
        for a in range(m):
            add_scaled(product + i * m, si + a * m, gain[i * m + a], m)
    copy_values(gain, product, n * m)
    for i in range(n):
        x[i] += dot(gain + i * m, y, m)
    var ikh = work
    var ap = work + n * n
    for i in range(n):
        fill_values(ikh + i * n, 0.0, n)
        ikh[i * n + i] = 1.0
        for a in range(m):
            add_scaled(ikh + i * n, h + a * n, -gain[i * m + a], n)
    for i in range(n):
        fill_values(ap + i * n, 0.0, n)
        for a in range(n):
            add_scaled(ap + i * n, cov + a * n, ikh[i * n + a], n)
    for i in range(n):
        fill_values(product + i * m, 0.0, m)
        for a in range(m):
            add_scaled(product + i * m, r + a * m, gain[i * m + a], m)
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
    k: Int,
    n: Int,
) abi("C"):
    var sigmas = p(sigmas_addr)
    var wm = p(wm_addr)
    var wc = p(wc_addr)
    var noise = p(noise_addr)
    var mean = p(mean_addr)
    var cov = p(cov_addr)
    fill_values(mean, 0.0, n)
    for i in range(k):
        add_scaled(mean, sigmas + i * n, wm[i], n)

    def covariance_row(i: Int) capturing:
        copy_values(cov + i * n, noise + i * n, n)
        for sidx in range(k):
            var scale = wc[sidx] * (sigmas[sidx * n + i] - mean[i])
            var j = 0
            var scales = SIMD[DType.float64, W](scale)
            while j + W <= n:
                cov.store(
                    i * n + j,
                    cov.load[width=W](i * n + j)
                    + scales
                    * (
                        sigmas.load[width=W](sidx * n + j)
                        - mean.load[width=W](j)
                    ),
                )
                j += W
            while j < n:
                cov[i * n + j] += scale * (sigmas[sidx * n + j] - mean[j])
                j += 1

    if k * n * n >= 1000000:
        parallelize[covariance_row](n, 8)
    else:
        for i in range(n):
            covariance_row(i)


def ut_mean_gpu(sigmas: Ptr, wm: Ptr, mean: Ptr, k: Int, n: Int):
    var j = Int(global_idx.x)
    if j < n:
        var acc = 0.0
        for i in range(k):
            acc += wm[i] * sigmas[i * n + j]
        mean[j] = acc


def ut_cov_gpu(
    sigmas: Ptr,
    wc: Ptr,
    noise: Ptr,
    mean: Ptr,
    cov: Ptr,
    k: Int,
    n: Int,
):
    var index = Int(global_idx.x)
    if index < n * n:
        var row = index // n
        var col = index - row * n
        var acc = noise[index]
        for sidx in range(k):
            acc += wc[sidx] * (sigmas[sidx * n + row] - mean[row]) * (
                sigmas[sidx * n + col] - mean[col]
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
    k: Int,
    n: Int,
) abi("C") -> Int:
    if k * n + 2 * k + 2 * n * n + n > 240000000:
        mfp_unscented_transform(
            sigmas_addr,
            wm_addr,
            wc_addr,
            noise_addr,
            mean_addr,
            cov_addr,
            k,
            n,
        )
        return 0
    try:
        var sigmas = p(sigmas_addr)
        var wm = p(wm_addr)
        var wc = p(wc_addr)
        var noise = p(noise_addr)
        var mean = p(mean_addr)
        var cov = p(cov_addr)
        var ctx = DeviceContext()
        if ctx.api() == "cpu":
            mfp_unscented_transform(
                sigmas_addr,
                wm_addr,
                wc_addr,
                noise_addr,
                mean_addr,
                cov_addr,
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
            k,
            n,
            grid_dim=(n + block_size - 1) // block_size,
            block_dim=block_size,
        )
        ctx.enqueue_function[ut_cov_gpu](
            d_sigmas,
            d_wc,
            d_noise,
            d_mean,
            d_cov,
            k,
            n,
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
    cumulative_addr: Int, positions_addr: Int, indexes_addr: Int, n: Int
) abi("C"):
    var cumulative = p(cumulative_addr)
    var positions = p(positions_addr)
    var indexes = ip(indexes_addr)

    def search(i: Int) capturing:
        var lo = 0
        var hi = n
        while lo < hi:
            var mid = (lo + hi) // 2
            if positions[i] < cumulative[mid]:
                hi = mid
            else:
                lo = mid + 1
        indexes[i] = Int64(min(lo, n - 1))

    if n >= 32768:
        comptime chunk_size = 4096
        var chunks = (n + chunk_size - 1) // chunk_size

        def search_chunk(chunk: Int) capturing:
            var begin = chunk * chunk_size
            var end = min(begin + chunk_size, n)
            for i in range(begin, end):
                var lo = 0
                var hi = n
                while lo < hi:
                    var mid = (lo + hi) // 2
                    if positions[i] < cumulative[mid]:
                        hi = mid
                    else:
                        lo = mid + 1
                indexes[i] = Int64(min(lo, n - 1))

        parallelize[search_chunk](chunks, 8)
    else:
        for i in range(n):
            search(i)


@export("mfp_neff")
def mfp_neff(weights_addr: Int, n: Int) abi("C") -> Float64:
    var weights = p(weights_addr)
    var acc = 0.0
    for i in range(n):
        acc += weights[i] * weights[i]
    return 1.0 / acc
