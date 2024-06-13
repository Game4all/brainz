const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const ops = @import("ops.zig");

/// Represents an activation function.
pub const Activation = struct {
    /// Applies the activation function to the input.
    apply: *const fn (*const Matrix(f32), *Matrix(f32)) *Matrix(f32),
    ///Applies the derivative of the activation function w.r.t the inputs.
    apply_derivative: *const fn (*const Matrix(f32), *Matrix(f32)) *Matrix(f32),
    /// Name of this activation function. Used for display purposes.
    name: [:0]const u8,
};

/// Linear activation function
pub const Linear: Activation = .{
    .apply = linear_activation,
    .apply_derivative = linear_derivative,
    .name = @tagName(.linear),
};

fn linear_activation(in: *const Matrix(f32), _: *Matrix(f32)) *Matrix(f32) {
    return @constCast(in);
}

fn linear_derivative(_: *const Matrix(f32), out: *Matrix(f32)) *Matrix(f32) {
    out.fill(1.0);
    return out;
}

/// REctified Linear Unit (ReLu) activation function
pub const ReLu: Activation = .{
    .apply = relu_activation,
    .apply_derivative = relu_derivative,
    .name = @tagName(.relu),
};

fn relu_activation(in: *const Matrix(f32), out: *Matrix(f32)) *Matrix(f32) {
    for (in.get_slice(), out.get_mut_slice()) |v, *r|
        r.* = @max(0, v);
    return out;
}

fn relu_derivative(in: *const Matrix(f32), out: *Matrix(f32)) *Matrix(f32) {
    for (in.get_slice(), out.get_mut_slice()) |i, *o|
        o.* = @max(0, std.math.sign(i));
    return out;
}

/// Sigmoid activation function
pub const Sigmoid: Activation = .{
    .apply = sigmoid_activation,
    .apply_derivative = sigmoid_derivative,
    .name = @tagName(.sigmoid),
};

fn sigmoid_activation(in: *const Matrix(f32), out: *Matrix(f32)) *Matrix(f32) {
    for (in.get_slice(), out.get_mut_slice()) |v, *r|
        r.* = 1.0 / (1.0 + std.math.exp(-v));
    return out;
}

fn sigmoid_derivative(in: *const Matrix(f32), out: *Matrix(f32)) *Matrix(f32) {
    for (in.get_slice(), out.get_mut_slice()) |i, *o| {
        const A = (1.0 / (1.0 + std.math.exp(-i)));
        o.* = A * (1.0 - A);
    }
    return out;
}

/// Heaviside aka step unit activation function
pub const Heaviside: Activation = .{
    .apply = heaviside_activation,
    .apply_derivative = heaviside_derivative,
    .name = @tagName(.heaviside),
};

fn heaviside_activation(in: *const Matrix(f32), out: *Matrix(f32)) *Matrix(f32) {
    for (in.get_slice(), out.get_mut_slice()) |i, *o| {
        o.* = @max(std.math.sign(i), 0);
    }
    return out;
}

fn heaviside_derivative(_: *const Matrix(f32), out: *Matrix(f32)) *Matrix(f32) {
    out.fill(1.0);
    return out;
}

/// Softmax activation function
pub const Softmax: Activation = .{
    .apply = softmax_activation,
    .apply_derivative = softmax_derivative,
    .name = @tagName(.softmax),
};

fn softmax_activation(in: *const Matrix(f32), out: *Matrix(f32)) *Matrix(f32) {
    ops.exp(f32, in, out);
    const s = ops.sum(f32, out);
    for (out.get_mut_slice()) |*v|
        v.* = v.* / s;
    return out;
}

fn softmax_derivative(in: *const Matrix(f32), out: *Matrix(f32)) *Matrix(f32) {
    // return in * (1.0 - in);
    for (in.get_slice(), out.get_mut_slice()) |i, *o| {
        const A = (1.0 / (1.0 + std.math.exp(-i)));
        o.* = A * (1.0 - A);
    }
    return out;
}

test "softmax activation" {
    var test_mat = try Matrix(f32).empty(.{ 3, 1 }, std.testing.allocator);
    defer test_mat.deinit();

    var softmax_mat = try Matrix(f32).empty(test_mat.shape, std.testing.allocator);
    defer softmax_mat.deinit();

    test_mat.set(.{ 0, 0 }, 1.0);
    test_mat.set(.{ 1, 0 }, 3.0);
    test_mat.set(.{ 2, 0 }, 6.0);

    _ = Softmax.apply(&test_mat, &softmax_mat);
    try std.testing.expectEqual(1.0, ops.sum(f32, &softmax_mat));
}

test "sigmoid activation" {
    var test_mat = try Matrix(f32).with_value(.{ 3, 1 }, 1.0, std.testing.allocator);
    defer test_mat.deinit();

    var sigmoid_mat = try Matrix(f32).empty(test_mat.shape, std.testing.allocator);
    defer sigmoid_mat.deinit();

    _ = Sigmoid.apply(&test_mat, &sigmoid_mat);
    try std.testing.expectEqual(2.19317573589, ops.sum(f32, &sigmoid_mat));

    test_mat.fill(0.0);
    _ = Sigmoid.apply(&test_mat, &sigmoid_mat);
    try std.testing.expectEqual(1.5, ops.sum(f32, &sigmoid_mat));
}
