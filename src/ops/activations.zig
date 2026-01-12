const std = @import("std");
const tensor = @import("../tensor.zig");

const Dtype = tensor.Dtype;
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

fn dispatchActivationForward(comptime ty: type, in: *const Tensor, output: *const Tensor, comptime op: anytype) !void {
    const outSlice = output.slice(ty).?;
    const aSlice = in.slice(ty).?;

    for (outSlice, aSlice) |*v, a_val|
        v.* = op(a_val);
}

fn activationForward(inputs: []const *const Tensor, output: *const Tensor, comptime op: anytype) !void {
    if (!output.dtype.isFloatingPoint()) return error.UnsupportedDtype;

    switch (output.dtype) {
        .float32 => try dispatchActivationForward(f32, inputs[0], output, op),
        .float64 => try dispatchActivationForward(f64, inputs[0], output, op),
        else => return error.UnsupportedDtype,
    }
}

fn dispatchActivationBackward(comptime ty: type, in: *const Tensor, out: *const Tensor, gradOutput: *const Tensor, comptime gradOp: anytype) !void {
    const gradOutSlice = gradOutput.slice(ty).?;
    const outSlice = out.slice(ty).?;
    const aSlice = in.slice(ty).?;

    const aGradSlice = if (in.grad) |g| g.slice(ty).? else return;

    for (aGradSlice, outSlice, gradOutSlice, aSlice) |*ga, o_val, gCommon, a_val|
        ga.* += gradOp(gCommon, a_val, o_val);
}

fn activationBackward(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, comptime gradOp: anytype) !void {
    const a = inputs[0];

    if (!gradOutput.dtype.isFloatingPoint()) return error.UnsupportedDtype;

    switch (gradOutput.dtype) {
        .float32 => try dispatchActivationBackward(f32, a, output, gradOutput, gradOp),
        .float64 => try dispatchActivationBackward(f64, a, output, gradOutput, gradOp),
        else => return error.UnsupportedDtype,
    }
}

// op functions
inline fn reluOp(a: anytype) @TypeOf(a) {
    return if (a > 0) a else 0;
}

inline fn sigmoidOp(a: anytype) @TypeOf(a) {
    return @as(@TypeOf(a), 1.0) / (1.0 + std.math.exp(-a));
}

// grad op functions (for backprop)
inline fn reluGradOp(gradOut: anytype, in: anytype, out: anytype) @TypeOf(gradOut) {
    _ = out;
    return if (in > 0) gradOut else 0;
}

inline fn sigmoidGradOp(gradOut: anytype, in: anytype, out: anytype) @TypeOf(gradOut) {
    _ = in;
    return gradOut * out * (1.0 - out);
}

// op callbacks
pub fn forwardReLU(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try activationForward(inputs, output, reluOp);
}

pub fn backwardReLU(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try activationBackward(inputs, output, gradOutput, reluGradOp);
}

pub fn forwardSigmoid(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try activationForward(inputs, output, sigmoidOp);
}

pub fn backwardSigmoid(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try activationBackward(inputs, output, gradOutput, sigmoidGradOp);
}

fn dispatchSoftmaxForward(comptime ty: type, in: *const Tensor, output: *const Tensor, axis: usize) void {
    const in_data = in.slice(ty).?;
    const out_data = output.slice(ty).?;

    const axis_dim = in.shape.dimensions[axis];
    const axis_stride = in.strides[axis];

    const total_elements = in.shape.totalLength();
    const n_slices = total_elements / axis_dim;

    var out_indices = std.mem.zeroes([Shape.MAX_DIMENSIONS]usize);
    for (0..n_slices) |_| {
        // calculate base offset for the slice
        var base_offset: usize = 0;
        var out_idx: usize = 0;
        for (0..in.shape.n_dimensions) |d| {
            if (d == axis) continue;
            base_offset += out_indices[out_idx] * in.strides[d];
            out_idx += 1;
        }

        // find max value along the axis
        var max_val = in_data[base_offset];
        for (1..axis_dim) |i| {
            const val = in_data[base_offset + i * axis_stride];
            if (val > max_val) max_val = val;
        }

        // perform exponentiation and sum all
        var sum: ty = 0;
        for (0..axis_dim) |i| {
            const val = @exp(in_data[base_offset + i * axis_stride] - max_val);
            out_data[base_offset + i * axis_stride] = val;
            sum += val;
        }

        // normalize
        const inv_sum = 1.0 / sum;
        for (0..axis_dim) |i| {
            out_data[base_offset + i * axis_stride] *= inv_sum;
        }

        // increment out_indices
        var d_idx: usize = in.shape.n_dimensions - 1;
        while (true) {
            if (d_idx == axis) {
                if (d_idx == 0) break;
                d_idx -= 1;
                continue;
            }
            out_indices[if (d_idx > axis) d_idx - 1 else d_idx] += 1;
            if (out_indices[if (d_idx > axis) d_idx - 1 else d_idx] < in.shape.dimensions[d_idx]) break;
            out_indices[if (d_idx > axis) d_idx - 1 else d_idx] = 0;
            if (d_idx == 0) break;
            d_idx -= 1;
        }
    }
}

pub fn forwardSoftmax(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    const in = inputs[0];
    const axis: usize = @intFromPtr(extraData);

    if (!output.dtype.isFloatingPoint()) return error.UnsupportedDtype;

    switch (output.dtype) {
        .float32 => dispatchSoftmaxForward(f32, in, output, axis),
        .float64 => dispatchSoftmaxForward(f64, in, output, axis),
        else => return error.UnsupportedDtype,
    }
}

fn dispatchSoftmaxBackward(comptime ty: type, in: *const Tensor, out: *const Tensor, gradOutput: *const Tensor, axis: usize) void {
    const out_data = out.slice(ty).?;
    const grad_out_data = gradOutput.slice(ty).?;
    const in_grad = in.grad.?.slice(ty).?;

    const axis_dim = in.shape.dimensions[axis];
    const axis_stride = in.strides[axis];

    const total_elements = in.shape.totalLength();
    const n_slices = total_elements / axis_dim;

    var out_indices = std.mem.zeroes([Shape.MAX_DIMENSIONS]usize);
    for (0..n_slices) |_| {
        var base_offset: usize = 0;
        var out_idx: usize = 0;
        for (0..in.shape.n_dimensions) |d| {
            if (d == axis) continue;
            base_offset += out_indices[out_idx] * in.strides[d];
            out_idx += 1;
        }

        // dL/dxi = yi * (dL/dyi - sum(dL/dyj * yj))
        var dot: ty = 0;
        for (0..axis_dim) |i| {
            const offset = base_offset + i * axis_stride;
            dot += grad_out_data[offset] * out_data[offset];
        }

        // compute gradien
        for (0..axis_dim) |i| {
            const offset = base_offset + i * axis_stride;
            in_grad[offset] += out_data[offset] * (grad_out_data[offset] - dot);
        }

        // increment out_indices
        var d_idx: usize = in.shape.n_dimensions - 1;
        while (true) {
            if (d_idx == axis) {
                if (d_idx == 0) break;
                d_idx -= 1;
                continue;
            }
            out_indices[if (d_idx > axis) d_idx - 1 else d_idx] += 1;
            if (out_indices[if (d_idx > axis) d_idx - 1 else d_idx] < in.shape.dimensions[d_idx]) break;
            out_indices[if (d_idx > axis) d_idx - 1 else d_idx] = 0;
            if (d_idx == 0) break;
            d_idx -= 1;
        }
    }
}

pub fn backwardSoftmax(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    const in = inputs[0];
    const axis: usize = @intFromPtr(extraData);

    if (in.grad == null) return;

    switch (gradOutput.dtype) {
        .float32 => dispatchSoftmaxBackward(f32, in, output, gradOutput, axis),
        .float64 => dispatchSoftmaxBackward(f64, in, output, gradOutput, axis),
        else => return error.UnsupportedDtype,
    }
}
