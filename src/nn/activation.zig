const std = @import("std");

/// Represents an activation function.
pub const Activation = struct {
    ///TODO: consider applying activation in bulk fashion to leverage SIMD optimizations.
    /// Applies the activation function to the input.
    apply: fn (f32, []f32) f32,
    ///Applies the derivative of the activation function w.r.t the output.
    apply_derivative: fn (f32, []f32) f32,
};

/// Linear activation function
pub const Linear: Activation = .{
    .apply = linear_activation,
    .apply_derivative = linear_derivative,
};

fn linear_activation(in: f32, _: []f32) f32 {
    return in;
}

fn linear_derivative(out: f32, _: []f32) f32 {
    return out;
}

/// REctified Linear Unit (ReLu) activation function
pub const ReLu: Activation = .{
    .apply = relu_activation,
    .apply_derivative = relu_derivative,
};

fn relu_activation(in: f32, _: []f32) f32 {
    return @max(0, in);
}

fn relu_derivative(out: f32, _: []f32) f32 {
    return out * @max(std.math.sign(out), 0);
}

/// Sigmoid activation function
pub const Sigmoid: Activation = .{
    .apply = sigmoid_activation,
    .apply_derivative = sigmoid_derivative,
};

fn sigmoid_activation(in: f32, _: []f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-in));
}

fn sigmoid_derivative(out: f32, _: []f32) f32 {
    return out * (1.0 - out);
}

/// Heaviside aka step unit activation function
pub const Heaviside: Activation = .{
    .apply = heaviside_activation,
    .apply_derivative = heaviside_derivative,
};

fn heaviside_activation(in: f32, _: []f32) f32 {
    return @max(std.math.sign(in), 0);
}

fn heaviside_derivative(out: f32, _: []f32) f32 {
    return @max(std.math.sign(out), 0) * out;
}

/// Softmax activation function
pub const Softmax: Activation = .{
    .apply = softmax_activation,
    .apply_derivative = softmax_derivative,
};

fn softmax_activation(in: f32, ins: []f32) f32 {
    var sum: f32 = 0.0;
    for (ins) |i|
        sum += std.math.exp(i);

    return std.math.exp(in) / sum;
}

fn softmax_derivative(in: f32, _: []f32) f32 {
    return in * (1.0 - in);
}
