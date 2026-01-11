const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

fn dispatchArgmaxForward(comptime ty: type, a: *const Tensor, output: *const Tensor, axis: usize) void {
    const a_data = a.slice(ty).?;
    const out_data = output.slice(u64).?;

    const a_strides = a.strides;
    const axis_dim = a.shape.dimensions[axis];
    const axis_stride = a_strides[axis];

    var out_indices = std.mem.zeroes([Shape.MAX_DIMENSIONS]usize);
    for (out_data) |*v| {
        // compute the base offset in the input for this output element
        var base_offset: usize = 0;
        var out_idx: usize = 0;
        for (0..a.shape.n_dimensions) |d| {
            if (d == axis) continue;
            // map output index to input dimension
            base_offset += out_indices[out_idx] * a_strides[d];
            out_idx += 1;
        }

        var max_val = a_data[base_offset];
        var max_idx: u64 = 0;

        for (1..axis_dim) |i| {
            const current_val = a_data[base_offset + i * axis_stride];
            if (current_val > max_val) {
                max_val = current_val;
                max_idx = @intCast(i);
            }
        }
        v.* = max_idx;

        // increment out_indices for next iteration
        var d: usize = output.shape.n_dimensions;
        while (d > 0) {
            d -= 1;
            out_indices[d] += 1;
            if (out_indices[d] < output.shape.dimensions[d]) break;
            out_indices[d] = 0;
        }
    }
}

pub fn forwardArgmax(inputs: []const *const Tensor, output: *const Tensor, extra: ?*anyopaque) !void {
    const a = inputs[0];
    const axis: usize = @intFromPtr(extra);

    switch (a.dtype) {
        .float32 => dispatchArgmaxForward(f32, a, output, axis),
        .float64 => dispatchArgmaxForward(f64, a, output, axis),
        else => @panic("Unsupported dtype for argmax"),
    }
}

// this is a no-op since argmax is not differentiable
pub fn backwardArgmax(inputs: []const *const Tensor, output: *const Tensor, grad_out: *const Tensor, extra: ?*anyopaque) !void {
    _ = inputs;
    _ = output;
    _ = grad_out;
    _ = extra;
}
