const std = @import("std");
const tensor = @import("../tensor.zig");

const Dtype = tensor.Dtype;
const Tensor = tensor.Tensor;
const Shape = tensor.Shape;

fn dispatchActivationForward(comptime ty: type, a: *const Tensor, output: *const Tensor, comptime op: anytype) !void {
    const outSlice = output.slice(ty).?;
    const aSlice = a.slice(ty).?;

    for (outSlice, aSlice) |*v, a_val| {
        v.* = op(a_val);
    }
}

fn activationForward(inputs: []const *const Tensor, output: *const Tensor, comptime op: anytype) !void {
    const a = inputs[0];

    if (!output.dtype.isFloatingPoint()) return error.UnsupportedDtype;

    switch (output.dtype) {
        .float32 => try dispatchActivationForward(f32, a, output, op),
        .float64 => try dispatchActivationForward(f64, a, output, op),
        else => {},
    }
}

fn dispatchActivationBackward(comptime ty: type, a: *const Tensor, gradOutput: *const Tensor, comptime gradOp: anytype) !void {
    const gradOutSlice = gradOutput.slice(ty).?;
    const aSlice = a.slice(ty).?;

    const aGradSlice = if (a.grad) |g| g.slice(ty) else null;
    if (aGradSlice == null) return;

    for (aGradSlice.?, gradOutSlice, aSlice) |*ga, gCommon, a_val| {
        ga.* += gradOp(gCommon, a_val);
    }
}

fn activationBackward(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, comptime gradOp: anytype) !void {
    _ = output;
    const a = inputs[0];

    if (!gradOutput.dtype.isFloatingPoint()) return error.UnsupportedDtype;

    switch (gradOutput.dtype) {
        .float32 => try dispatchActivationBackward(f32, a, gradOutput, gradOp),
        .float64 => try dispatchActivationBackward(f64, a, gradOutput, gradOp),
        else => {},
    }
}

// op functions
inline fn reluOp(a: anytype) @TypeOf(a) {
    return if (a > 0) a else 0;
}

// grad op functions (for backprop)
inline fn reluGradOp(gradOut: anytype, a: anytype) @TypeOf(gradOut) {
    return if (a > 0) gradOut else 0;
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
