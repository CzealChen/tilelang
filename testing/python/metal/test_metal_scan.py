import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def scan_kernel(n, dtype="float32", threads=128, reverse=False, inplace=True, scope="shared", repeats=1):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=threads):
            source = T.alloc_shared((n,), dtype, scope=scope)
            dest = T.alloc_shared((n,), dtype, scope=scope)
            T.copy(A, source)
            for _ in T.serial(repeats):
                if inplace:
                    T.cumsum(source, reverse=reverse)
                else:
                    T.cumsum(source, dest, reverse=reverse)
                    T.copy(dest, source)
            T.copy(source, B)

    return main


def compile_metal(program):
    return tilelang.compile(program, target="metal", target_host="c", execution_backend="tvm_ffi", out_idx=[])


def lower_metal(program):
    with tilelang.tvm.target.Target("metal"):
        return tilelang.lower(program, target="metal", enable_host_codegen=False, enable_device_compile=False)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "n,dtype,threads,reverse,inplace,scope,repeats",
    [
        (1, "int32", 32, False, True, "shared", 1),
        (17, "int32", 128, True, False, "shared", 1),
        (32, "float32", 32, False, True, "shared", 1),
        (33, "int32", 128, False, True, "shared.dyn", 2),
        (65, "float32", 128, True, True, "shared", 2),
        (1025, "float32", 128, False, False, "shared.dyn", 1),
    ],
)
def test_metal_cumsum(n, dtype, threads, reverse, inplace, scope, repeats):
    kernel = compile_metal(scan_kernel(n, dtype, threads, reverse, inplace, scope, repeats))
    torch.manual_seed(0)
    x = torch.randint(-4, 5, (n,), dtype=torch.int32).to(getattr(torch, dtype))
    if dtype == "float32":
        x = x / 7
    expected = x
    for _ in range(repeats):
        ordered = expected.flip([0]) if reverse else expected
        expected = ordered.cumsum(0, dtype=x.dtype)
        if reverse:
            expected = expected.flip([0])
    output = torch.empty_like(x, device="mps")
    device_input = x.to("mps")
    kernel(device_input, output)
    torch.testing.assert_close(output.cpu(), expected, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(device_input.cpu(), x)


def region_kernel(src_begin, dst_begin, n=33, dtype="int32", reverse=False):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), B: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), dtype)
            T.copy(A, shared)
            T.cumsum(shared[src_begin : src_begin + n], shared[dst_begin : dst_begin + n], reverse=reverse)
            T.copy(shared, B)

    return main


@tilelang.testing.requires_metal
@pytest.mark.parametrize("src_begin,dst_begin,reverse", [(3, 3, True), (3, 36, False), (66, 5, True)])
def test_metal_cumsum_regions(src_begin, dst_begin, reverse):
    kernel = compile_metal(region_kernel(src_begin, dst_begin, reverse=reverse))
    x = torch.arange(128, dtype=torch.int32) % 11 - 5
    expected = x.clone()
    values = x[src_begin : src_begin + 33]
    result = (values.flip([0]) if reverse else values).cumsum(0, dtype=torch.int32)
    expected[dst_begin : dst_begin + 33] = result.flip([0]) if reverse else result
    output = torch.empty_like(x, device="mps")
    kernel(x.to("mps"), output)
    torch.testing.assert_close(output.cpu(), expected)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("src_begin,dst_begin", [(3, 4), (4, 3)])
def test_metal_cumsum_rejects_overlapping_regions(src_begin, dst_begin):
    with pytest.raises(Exception, match="identical or disjoint regions"):
        lower_metal(region_kernel(src_begin, dst_begin))


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype,threads,message", [("float16", 32, "int32 or float32"), ("int32", 16, "thread extent")])
def test_metal_cumsum_unsupported(dtype, threads, message):
    with pytest.raises(Exception, match=message):
        lower_metal(scan_kernel(33, dtype=dtype, threads=threads))


@tilelang.testing.requires_metal
def test_metal_cumsum_fragment_staging():
    @T.prim_func
    def main(A: T.Tensor((65,), "float32"), B: T.Tensor((65,), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((65,), "float32")
            T.copy(A, fragment)
            T.cumsum(fragment, dim=-1, reverse=True)
            T.copy(fragment, B)

    kernel = compile_metal(main)
    x = torch.arange(65, dtype=torch.float32) / 17 - 2
    output = torch.empty_like(x, device="mps")
    kernel(x.to("mps"), output)
    torch.testing.assert_close(output.cpu(), x.flip([0]).cumsum(0).flip([0]), atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("guard", ["warp", "lanes"])
def test_metal_cumsum_rejects_divergent_calls(guard):
    @T.prim_func
    def main(A: T.Tensor((64,), "int32"), B: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64,), "int32")
            T.copy(A, shared)
            tx = T.get_thread_binding(0)
            if guard == "warp":
                if tx < 32:
                    T.cumsum(shared)
            else:
                if tx % 2 == 0:
                    T.cumsum(shared)
            T.copy(shared, B)

    with pytest.raises(Exception, match="block-uniform control flow"):
        lower_metal(main)


@tilelang.testing.requires_metal
def test_metal_cumsum_rejects_multidimensional_threadgroup():
    with pytest.raises(Exception, match="one-dimensional threadgroup"):
        lower_metal(scan_kernel(64, threads=(32, 2)))


@tilelang.testing.requires_metal
def test_metal_cumsum_block_uniform_condition():
    @T.prim_func
    def main(A: T.Tensor((2, 65), "int32"), B: T.Tensor((2, 65), "int32")):
        with T.Kernel(2, threads=128) as bx:
            shared = T.alloc_shared((65,), "int32")
            T.copy(A[bx, 0:65], shared)
            if bx == 0:
                T.cumsum(shared)
            T.copy(shared, B[bx, 0:65])

    kernel = compile_metal(main)
    x = torch.arange(130, dtype=torch.int32).reshape(2, 65) % 7 - 3
    expected = x.clone()
    expected[0] = x[0].cumsum(0, dtype=torch.int32)
    output = torch.empty_like(x, device="mps")
    kernel(x.to("mps"), output)
    torch.testing.assert_close(output.cpu(), expected)


@tilelang.testing.requires_metal
def test_metal_cumsum_rejects_thread_dependent_region():
    @T.prim_func
    def main(A: T.Tensor((256,), "int32"), B: T.Tensor((256,), "int32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((256,), "int32")
            T.copy(A, shared)
            tx = T.get_thread_binding(0)
            T.cumsum(shared[tx : tx + 33])
            T.copy(shared, B)

    with pytest.raises(Exception, match="block-uniform region offsets"):
        lower_metal(main)


@tilelang.testing.requires_metal
def test_metal_cumsum_block_uniform_region():
    @T.prim_func
    def main(A: T.Tensor((2, 128), "int32"), B: T.Tensor((2, 128), "int32")):
        with T.Kernel(2, threads=128) as bx:
            shared = T.alloc_shared((128,), "int32")
            T.copy(A[bx, 0:128], shared)
            T.cumsum(shared[bx * 33 : bx * 33 + 33])
            T.copy(shared, B[bx, 0:128])

    kernel = compile_metal(main)
    x = torch.arange(256, dtype=torch.int32).reshape(2, 128) % 11 - 5
    expected = x.clone()
    for row in range(2):
        expected[row, row * 33 : row * 33 + 33] = x[row, row * 33 : row * 33 + 33].cumsum(0, dtype=torch.int32)
    output = torch.empty_like(x, device="mps")
    kernel(x.to("mps"), output)
    torch.testing.assert_close(output.cpu(), expected)


@tilelang.testing.requires_metal
def test_metal_cumsum_rejects_while_loop():
    @T.prim_func
    def main(A: T.Tensor((64,), "int32"), B: T.Tensor((64,), "int32"), count: T.int32):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64,), "int32")
            iteration = T.alloc_var("int32", init=0)
            T.copy(A, shared)
            while iteration < count:
                T.cumsum(shared)
                iteration += 1
            T.copy(shared, B)

    with pytest.raises(Exception, match="does not yet support while loops"):
        lower_metal(main)


@tilelang.testing.requires_metal
def test_metal_cumsum_rejects_dynamic_length():
    @T.prim_func
    def main(A: T.Tensor((64,), "int32"), B: T.Tensor((64,), "int32"), length: T.int32):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64,), "int32")
            T.copy(A, shared)
            T.cumsum(shared[0:length])
            T.copy(shared, B)

    with pytest.raises(Exception, match="positive constant region lengths"):
        lower_metal(main)


if __name__ == "__main__":
    tilelang.testing.main()
