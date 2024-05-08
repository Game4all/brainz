const std = @import("std");

// the minimum value that a f32 can handle
// this gets added to the input value in loss functions which aren't numerically stable (namely log()) to prevent NaNs.
const F32_MIN_VALUE = std.math.floatMin(f32);

/// A usable loss function.
pub const Loss = struct {
    compute: fn (f32, f32) f32,
    compute_derivative: fn (f32, f32) f32,
};

/// Mean Squared Error or L2 loss.
pub const MSE: Loss = .{
    .compute = mse_loss,
    .compute_derivative = mse_derivative,
};

fn mse_loss(input: f32, target: f32) f32 {
    return (1.0 / 2.0) * std.math.pow(f32, target - input, 2.0);
}

fn mse_derivative(input: f32, target: f32) f32 {
    return (target - input);
}

/// Binary cross entropy.
/// This loss function assumes the output layer uses sigmoid activation.
pub const BinaryCrossEntropy: Loss = .{
    .compute = bce_loss,
    .compute_derivative = bce_derivative,
};

fn bce_loss(input: f32, target: f32) f32 {
    return -((target * @log(input + F32_MIN_VALUE)) + (1.0 - target) * @log(1.0 - input + F32_MIN_VALUE));
}

fn bce_derivative(input: f32, target: f32) f32 {
    return -((input + F32_MIN_VALUE - target) / ((input + F32_MIN_VALUE) * (1.0 - input + F32_MIN_VALUE)));
}

/// Categorical cross entropy.
/// This loss function asumes the output layer uses softmax activation.
pub const CategoricalCrossEntropy: Loss = .{
    .compute = cce_loss,
    .compute_derivative = bce_derivative,
};

fn cce_loss(input: f32, target: f32) f32 {
    return -target * @log(input + F32_MIN_VALUE);
}

fn cce_derivative(input: f32, target: f32) f32 {
    return input - target;
}
