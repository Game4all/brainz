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
