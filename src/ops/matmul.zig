const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;

// (M, N) x (N, K) -> (M, K)
pub fn forwardMatMul(inputs: []const *const Tensor, out: *const Tensor, extra: ?*anyopaque) !void {
    _ = extra;
    const a = inputs[0];
    const b = inputs[1];

    const M = a.shape.dimensions[0];
    const N = a.shape.dimensions[1]; // should be equal to b.shape.dimensions[0]
    const K = b.shape.dimensions[1];

    switch (out.dtype) {
        .float32 => {
            const out_data = out.slice(f32).?;
            const a_data = a.slice(f32).?;
            const b_data = b.slice(f32).?;

            // Initialize output to 0
            @memset(out_data, 0);

            // Naive matrix multiplication
            // Optimize this later if needed
            for (0..M) |m| {
                for (0..K) |k| {
                    var sum: f32 = 0;
                    for (0..N) |n| {
                        sum += a_data[m * N + n] * b_data[n * K + k];
                    }
                    out_data[m * K + k] = sum;
                }
            }
        },
        else => @panic("Unsupported dtype for matmul"),
    }
}

pub fn backwardMatMul(inputs: []const *const Tensor, out: *const Tensor, grad_out: *const Tensor, extra: ?*anyopaque) !void {
    _ = out;
    _ = extra;
    const a = inputs[0];
    const b = inputs[1];

    const M = a.shape.dimensions[0];
    const N = a.shape.dimensions[1];
    const K = b.shape.dimensions[1];

    switch (grad_out.dtype) {
        .float32 => {
            const out_grad = grad_out.slice(f32).?;

            if (a.requires_grad) {
                const a_grad = a.grad.?.slice(f32).?;
                const b_data = b.slice(f32).?;
                // dA = dC * B^T
                // (M, K) * (K, N) -> (M, N)

                // For each element in A (m, n)
                for (0..M) |m| {
                    for (0..N) |n| {
                        var sum: f32 = 0;
                        for (0..K) |k| {
                            // dC[m, k] * B[n, k] (B Transposed means swapping indices)
                            // B is (N, K), so B[n, k] is at n*K + k
                            sum += out_grad[m * K + k] * b_data[n * K + k];
                        }
                        a_grad[m * N + n] += sum; // Accumulate gradient
                    }
                }
            }

            if (b.requires_grad) {
                const b_grad = b.grad.?.slice(f32).?;
                const a_data = a.slice(f32).?;
                // dB = A^T * dC
                // (N, M) * (M, K) -> (N, K)

                // For each element in B (n, k)
                for (0..N) |n| {
                    for (0..K) |k| {
                        var sum: f32 = 0;
                        for (0..M) |m| {
                            // A[m, n] * dC[m, k]
                            // A is (M, N), so A[m, n] is at m*N + n
                            sum += a_data[m * N + n] * out_grad[m * K + k];
                        }
                        b_grad[n * K + k] += sum; // Accumulate gradient
                    }
                }
            }
        },
        else => @panic("Unsupported dtype for matmul backward"),
    }
}

// (B, M, N) x (N, K) -> (B, M, K)
pub fn forwardBatchedMatMul(inputs: []const *const Tensor, out: *const Tensor, extra: ?*anyopaque) !void {
    _ = extra;
    const a = inputs[0];
    const b = inputs[1];

    const B = a.shape.dimensions[0];
    const M = a.shape.dimensions[1];
    const N = a.shape.dimensions[2]; // a is (B, M, N)
    const K = b.shape.dimensions[1]; // b is (N, K)

    switch (out.dtype) {
        .float32 => {
            const out_data = out.slice(f32).?;
            const a_data = a.slice(f32).?;
            const b_data = b.slice(f32).?;

            @memset(out_data, 0);

            for (0..B) |b_idx| {
                for (0..M) |m| {
                    for (0..K) |k| {
                        var sum: f32 = 0;
                        for (0..N) |n| {
                            sum += a_data[b_idx * M * N + m * N + n] * b_data[n * K + k];
                        }
                        out_data[b_idx * M * K + m * K + k] = sum;
                    }
                }
            }
        },
        else => @panic("Unsupported dtype for batched matmul"),
    }
}

pub fn backwardBatchedMatMul(inputs: []const *const Tensor, out: *const Tensor, grad_out: *const Tensor, extra: ?*anyopaque) !void {
    _ = out;
    _ = extra;
    const a = inputs[0];
    const b = inputs[1];

    const B = a.shape.dimensions[0];
    const M = a.shape.dimensions[1];
    const N = a.shape.dimensions[2];
    const K = b.shape.dimensions[1];

    switch (grad_out.dtype) {
        .float32 => {
            const out_grad = grad_out.slice(f32).?;

            if (a.requires_grad) {
                const a_grad = a.grad.?.slice(f32).?;
                const b_data = b.slice(f32).?;

                // dA[b, m, n] = sum_k (dOut[b, m, k] * B[n, k])
                for (0..B) |b_idx| {
                    for (0..M) |m| {
                        for (0..N) |n| {
                            var sum: f32 = 0;
                            for (0..K) |k| {
                                sum += out_grad[b_idx * M * K + m * K + k] * b_data[n * K + k];
                            }
                            a_grad[b_idx * M * N + m * N + n] += sum;
                        }
                    }
                }
            }

            if (b.requires_grad) {
                const b_grad = b.grad.?.slice(f32).?;
                const a_data = a.slice(f32).?;

                // dB[n, k] = sum_b,m (dOut[b, m, k] * A[b, m, n])
                for (0..N) |n| {
                    for (0..K) |k| {
                        var sum: f32 = 0;
                        for (0..B) |b_idx| {
                            for (0..M) |m| {
                                sum += out_grad[b_idx * M * K + m * K + k] * a_data[b_idx * M * N + m * N + n];
                            }
                        }
                        b_grad[n * K + k] += sum;
                    }
                }
            }
        },
        else => @panic("Unsupported dtype for batched matmul backward"),
    }
}
