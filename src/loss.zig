const std = @import("std");
const ops = @import("ops.zig");
const Tensor = @import("tensor.zig").Tensor;

// the minimum value that a f32 can handle
// this gets added to the input value in loss functions which aren't numerically stable (namely log()) to prevent NaNs.
const F32_MIN_VALUE = std.math.floatMin(f32);

/// A usable loss function.
pub const Loss = struct {
    compute: *const fn (*const Tensor(f32), *const Tensor(f32)) f32,
    computeDerivative: *const fn (*const Tensor(f32), *const Tensor(f32), *Tensor(f32)) void,
};

/// Mean Squared Error or L2 loss.
pub const MSE: Loss = .{
    .compute = mse_loss,
    .computeDerivative = mse_derivative,
};

fn mse_loss(input: *const Tensor(f32), target: *const Tensor(f32)) f32 {
    var loss: f32 = 0.0;
    const len = input.constSlice().len;

    for (input.constSlice(), target.constSlice()) |i, t|
        loss += (1.0 / @as(f32, @floatFromInt(len))) * std.math.pow(f32, t - i, 2.0);

    return loss;
}

fn mse_derivative(in: *const Tensor(f32), target: *const Tensor(f32), result: *Tensor(f32)) void {
    ops.sub(f32, in, target, result);
    return;
}

/// Binary cross entropy.
/// This loss function assumes the output layer uses sigmoid activation.
pub const BinaryCrossEntropy: Loss = .{
    .compute = bce_loss,
    .computeDerivative = bce_derivative,
};

fn bce_loss(input: *const Tensor(f32), target: *const Tensor(f32)) f32 {
    var loss: f32 = 0.0;
    const len = input.constSlice().len;

    for (input.constSlice(), target.constSlice()) |i, t|
        loss += -(1.0 / @as(f32, @floatFromInt(len))) * ((t * @log(i + F32_MIN_VALUE)) + (1.0 - t) * @log(1.0 - i + F32_MIN_VALUE));

    return loss;
}

fn bce_derivative(input: *const Tensor(f32), target: *const Tensor(f32), result: *Tensor(f32)) void {
    for (input.constSlice(), target.constSlice(), result.slice()) |i, t, *r|
        r.* = ((i + F32_MIN_VALUE - t) / ((i + F32_MIN_VALUE) * (1.0 - i + F32_MIN_VALUE)));
}

/// Categorical cross entropy.
/// This loss function asumes the output layer uses softmax activation.
pub const CategoricalCrossEntropy: Loss = .{
    .compute = cce_loss,
    .computeDerivative = cce_derivative,
};

fn cce_loss(input: *const Tensor(f32), target: *const Tensor(f32)) f32 {
    var loss: f32 = 0.0;
    const len = input.constSlice().len;

    for (input.constSlice(), target.constSlice()) |i, t|
        loss += -(1.0 / @as(f32, @floatFromInt(len))) * t * @log(i + F32_MIN_VALUE);

    return loss;
}

fn cce_derivative(in: *const Tensor(f32), target: *const Tensor(f32), result: *Tensor(f32)) void {
    ops.sub(f32, in, target, result);
}
