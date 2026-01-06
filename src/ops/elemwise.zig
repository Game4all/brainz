const std = @import("std");
const tensor = @import("../tensor.zig");

const Dtype = tensor.Dtype;
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

fn dispatchElemwiseForward(comptime ty: type, a: *const Tensor, a_strides: []const usize, b: *const Tensor, b_strides: []const usize, output: *const Tensor, comptime op: anytype) !void {
    const outSlice = output.slice(ty).?;
    const aSlice = a.slice(ty).?;
    const bSlice = b.slice(ty).?;

    // we need to keep track of the indices of the output tensor to access the correct elements of the input tensors
    var indices = std.mem.zeroes([Shape.MAX_DIMENSIONS]usize);
    for (outSlice) |*v| {
        var a_offset: usize = 0;
        var b_offset: usize = 0;
        for (0..output.shape.n_dimensions) |d| {
            a_offset += indices[d] * a_strides[d];
            b_offset += indices[d] * b_strides[d];
        }

        v.* = op(aSlice[a_offset], bSlice[b_offset]);

        // increment indices for next iteration
        var d: usize = output.shape.n_dimensions;
        while (d > 0) {
            d -= 1;
            indices[d] += 1;
            if (indices[d] < output.shape.dimensions[d]) break;
            indices[d] = 0;
        }
    }
}

//TODO: with contiguous storage, this could be SIMD-accelerated pretty easily
/// Performs a generic, element-wise operation on two tensors writing the result to an output tensor.
fn elementWiseForward(inputs: []const *const Tensor, output: *const Tensor, comptime op: anytype) !void {
    const a = inputs[0];
    const b = inputs[1];

    if (!output.dtype.isFloatingPoint()) return error.UnsupportedDtype;

    const a_strides = a.shape.broadcastStrides(output.shape);
    const b_strides = b.shape.broadcastStrides(output.shape);

    switch (output.dtype) {
        .float32 => try dispatchElemwiseForward(f32, a, &a_strides, b, &b_strides, output, op),
        .float64 => try dispatchElemwiseForward(f64, a, &a_strides, b, &b_strides, output, op),
        else => {},
    }
}

fn dispatchElemwiseBackward(comptime ty: type, a: *const Tensor, a_strides: []const usize, b: *const Tensor, b_strides: []const usize, gradOutput: *const Tensor, comptime gradOp: anytype) !void {
    const gradOutSlice = gradOutput.slice(ty).?;

    // the computation of some gradients require having both inputs at hand
    const aSlice = a.slice(ty).?;
    const bSlice = b.slice(ty).?;

    const aGradSlice = if (a.grad) |g| g.slice(ty) else null;
    const bGradSlice = if (b.grad) |g| g.slice(ty) else null;

    if (aGradSlice == null and bGradSlice == null) return;

    // we need to keep track of the indices of the output tensor to access the correct elements of the input tensors
    var indices = std.mem.zeroes([tensor.Shape.MAX_DIMENSIONS]usize);
    for (gradOutSlice) |gCommon| {
        var a_offset: usize = 0;
        var b_offset: usize = 0;
        for (0..gradOutput.shape.n_dimensions) |d| {
            a_offset += indices[d] * a_strides[d];
            b_offset += indices[d] * b_strides[d];
        }

        const grads = gradOp(gCommon, aSlice[a_offset], bSlice[b_offset]);

        if (aGradSlice) |gs| gs[a_offset] += grads.da;
        if (bGradSlice) |gs| gs[b_offset] += grads.db;

        // increment indices for next iteration
        var d: usize = gradOutput.shape.n_dimensions;
        while (d > 0) {
            d -= 1;
            indices[d] += 1;
            if (indices[d] < gradOutput.shape.dimensions[d]) break;
            indices[d] = 0;
        }
    }
}

//TODO: with contiguous storage, this could be SIMD-accelerated pretty easily
/// Performs a generic, element-wise backward pass writing the results to the gradients of the input tensors.
fn elementWiseBackward(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, comptime gradOp: anytype) !void {
    _ = output;
    const a = inputs[0];
    const b = inputs[1];

    if (!gradOutput.dtype.isFloatingPoint()) return error.UnsupportedDtype;

    const a_strides = a.shape.broadcastStrides(gradOutput.shape);
    const b_strides = b.shape.broadcastStrides(gradOutput.shape);

    switch (gradOutput.dtype) {
        .float32 => try dispatchElemwiseBackward(f32, a, &a_strides, b, &b_strides, gradOutput, gradOp),
        .float64 => try dispatchElemwiseBackward(f64, a, &a_strides, b, &b_strides, gradOutput, gradOp),
        else => {},
    }
}

// Op functions
inline fn addOp(a: anytype, b: anytype) @TypeOf(a) {
    return a + b;
}
inline fn subOp(a: anytype, b: anytype) @TypeOf(a) {
    return a - b;
}
inline fn mulOp(a: anytype, b: anytype) @TypeOf(a) {
    return a * b;
}
inline fn divOp(a: anytype, b: anytype) @TypeOf(a) {
    return a / b;
}

// Grad Op functions (Backward)
// Return the gradients of both inputs
inline fn addGradOp(gradOut: anytype, a: anytype, b: anytype) struct { da: @TypeOf(gradOut), db: @TypeOf(gradOut) } {
    _ = a;
    _ = b;
    return .{ .da = gradOut, .db = gradOut };
}

inline fn subGradOp(gradOut: anytype, a: anytype, b: anytype) struct { da: @TypeOf(gradOut), db: @TypeOf(gradOut) } {
    _ = a;
    _ = b;
    return .{ .da = gradOut, .db = -gradOut };
}

inline fn mulGradOp(gradOut: anytype, a: anytype, b: anytype) struct { da: @TypeOf(gradOut), db: @TypeOf(gradOut) } {
    return .{ .da = gradOut * b, .db = gradOut * a };
}

inline fn divGradOp(gradOut: anytype, a: anytype, b: anytype) struct { da: @TypeOf(gradOut), db: @TypeOf(gradOut) } {
    const invB = 1.0 / b;
    return .{ .da = gradOut * invB, .db = gradOut * (-a * (invB * invB)) };
}

// Add operation

pub fn forwardAdd(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, addOp);
}

pub fn backwardAdd(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, addGradOp);
}

// Sub operation
pub fn forwardSub(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, subOp);
}

pub fn backwardSub(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, subGradOp);
}

// Mul operation
pub fn forwardMul(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, mulOp);
}

pub fn backwardMul(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, mulGradOp);
}

// Div operation
pub fn forwardDiv(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, divOp);
}

pub fn backwardDiv(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, divGradOp);
}
