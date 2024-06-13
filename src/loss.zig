const std = @import("std");
const ops = @import("ops.zig");
const Matrix = @import("matrix.zig").Matrix;

// the minimum value that a f32 can handle
// this gets added to the input value in loss functions which aren't numerically stable (namely log()) to prevent NaNs.
const F32_MIN_VALUE = std.math.floatMin(f32);

/// A usable loss function.
pub const Loss = struct {
    compute: *const fn (*const Matrix(f32), *const Matrix(f32)) f32,
    compute_derivative: *const fn (*const Matrix(f32), *const Matrix(f32), *Matrix(f32)) void,
};

/// Mean Squared Error or L2 loss.
pub const MSE: Loss = .{
    .compute = mse_loss,
    .compute_derivative = mse_derivative,
};

fn mse_loss(input: *const Matrix(f32), target: *const Matrix(f32)) f32 {
    var loss: f32 = 0.0;
    const len = input.get_slice().len;

    for (input.get_slice(), target.get_slice()) |i, t|
        loss += (1.0 / @as(f32, @floatFromInt(len))) * std.math.pow(f32, t - i, 2.0);

    return loss;
}

fn mse_derivative(in: *const Matrix(f32), target: *const Matrix(f32), result: *Matrix(f32)) void {
    ops.sub(f32, in, target, result);
    return;
}

/// Binary cross entropy.
/// This loss function assumes the output layer uses sigmoid activation.
pub const BinaryCrossEntropy: Loss = .{
    .compute = bce_loss,
    .compute_derivative = bce_derivative,
};

fn bce_loss(input: *const Matrix(f32), target: *const Matrix(f32)) f32 {
    var loss: f32 = 0.0;
    const len = input.get_slice().len;

    for (input.get_slice(), target.get_slice()) |i, t|
        loss += -(1.0 / @as(f32, @floatFromInt(len))) * ((t * @log(i + F32_MIN_VALUE)) + (1.0 - t) * @log(1.0 - i + F32_MIN_VALUE));

    return loss;
}

fn bce_derivative(input: *const Matrix(f32), target: *const Matrix(f32), result: *Matrix(f32)) void {
    for (input.get_slice(), target.get_slice(), result.get_mut_slice()) |i, t, *r|
        r.* = -((i + F32_MIN_VALUE - t) / ((i + F32_MIN_VALUE) * (1.0 - i + F32_MIN_VALUE)));

    return result;
}

/// Categorical cross entropy.
/// This loss function asumes the output layer uses softmax activation.
pub const CategoricalCrossEntropy: Loss = .{
    .compute = cce_loss,
    .compute_derivative = bce_derivative,
};

fn cce_loss(input: *const Matrix(f32), target: *const Matrix(f32)) f32 {
    var loss: f32 = 0.0;
    const len = input.get_slice().len;

    for (input.get_slice(), target.get_slice()) |i, t|
        loss += -(1.0 / @as(f32, @floatFromInt(len))) * t * @log(i + F32_MIN_VALUE);

    return loss;
}

fn cce_derivative(in: *const Matrix(f32), target: *const Matrix(f32), result: *Matrix(f32)) void {
    ops.sub(f32, in, target, result);
    return result;
}
