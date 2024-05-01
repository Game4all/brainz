const std = @import("std");

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
pub const BinaryCrossEntropy: Loss = .{
    .compute = bce_loss,
    .compute_derivative = bce_derivative,
};

fn bce_loss(input: f32, target: f32) f32 {
    return -((target * @log(input)) + (1.0 - target) * @log(1.0 - input));
}

fn bce_derivative(input: f32, target: f32) f32 {
    return -((input - target) / (input * (1.0 - input)));
}
