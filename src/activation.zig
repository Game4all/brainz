const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");

/// Represents an activation function.
pub const Activation = struct {
    /// Applies the activation function to the input.
    apply: *const fn (*const Tensor(f32), *Tensor(f32)) *Tensor(f32),
    ///Applies the derivative of the activation function w.r.t the inputs.
    applyDerivative: *const fn (*const Tensor(f32), *Tensor(f32)) *Tensor(f32),
    /// Name of this activation function. Used for display purposes.
    name: [:0]const u8,
};

/// Linear activation function
pub const Linear: Activation = .{
    .apply = linear_activation,
    .applyDerivative = linear_derivative,
    .name = @tagName(.linear),
};

fn linear_activation(in: *const Tensor(f32), _: *Tensor(f32)) *Tensor(f32) {
    return @constCast(in);
}

fn linear_derivative(_: *const Tensor(f32), out: *Tensor(f32)) *Tensor(f32) {
    out.fill(1.0);
    return out;
}

/// REctified Linear Unit (ReLu) activation function
pub const ReLu: Activation = .{
    .apply = relu_activation,
    .applyDerivative = relu_derivative,
    .name = @tagName(.relu),
};

fn relu_activation(in: *const Tensor(f32), out: *Tensor(f32)) *Tensor(f32) {
    ops.relu(f32, in, out);
    return out;
}

fn relu_derivative(in: *const Tensor(f32), out: *Tensor(f32)) *Tensor(f32) {
    ops.reluBackprop(f32, in, out);
    return out;
}

/// Sigmoid activation function
pub const Sigmoid: Activation = .{
    .apply = sigmoid_activation,
    .applyDerivative = sigmoid_derivative,
    .name = @tagName(.sigmoid),
};

fn sigmoid_activation(in: *const Tensor(f32), out: *Tensor(f32)) *Tensor(f32) {
    ops.sigmoid(f32, in, out);
    return out;
}

fn sigmoid_derivative(in: *const Tensor(f32), out: *Tensor(f32)) *Tensor(f32) {
    ops.sigmoidBackprop(f32, in, out);
    return out;
}

/// Heaviside aka step unit activation function
pub const Heaviside: Activation = .{
    .apply = heaviside_activation,
    .applyDerivative = heaviside_derivative,
    .name = @tagName(.heaviside),
};

fn heaviside_activation(in: *const Tensor(f32), out: *Tensor(f32)) *Tensor(f32) {
    for (in.constSlice(), out.slice()) |i, *o| {
        o.* = @max(std.math.sign(i), 0);
    }
    return out;
}

fn heaviside_derivative(_: *const Tensor(f32), out: *Tensor(f32)) *Tensor(f32) {
    out.fill(1.0);
    return out;
}

/// Softmax activation function
pub const Softmax: Activation = .{
    .apply = softmax_activation,
    .applyDerivative = softmax_derivative,
    .name = @tagName(.softmax),
};

fn softmax_activation(in: *const Tensor(f32), out: *Tensor(f32)) *Tensor(f32) {
    ops.exp(f32, in, out);
    const s = ops.sum(f32, out);
    ops.mulScalar(f32, out, 1.0 / s, out); // s can't be ever 0.0 since exp can't be 0.0.
    return out;
}

fn softmax_derivative(in: *const Tensor(f32), out: *Tensor(f32)) *Tensor(f32) {
    // return in * (1.0 - in);
    for (in.constSlice(), out.slice()) |i, *o| {
        const A = (1.0 / (1.0 + std.math.exp(-i)));
        o.* = A * (1.0 - A);
    }
    return out;
}

test "softmax activation" {
    var test_mat = try Tensor(f32).alloc(.{ 0, 3, 1 }, std.testing.allocator);
    defer test_mat.deinit(std.testing.allocator);

    var softmax_mat = try Tensor(f32).alloc(test_mat.shape, std.testing.allocator);
    defer softmax_mat.deinit(std.testing.allocator);

    test_mat.set(.{ 0, 0, 0 }, 1.0);
    test_mat.set(.{ 0, 1, 0 }, 3.0);
    test_mat.set(.{ 0, 2, 0 }, 6.0);

    _ = Softmax.apply(&test_mat, &softmax_mat);
    try std.testing.expectEqual(1.0, ops.sum(f32, &softmax_mat));
}

test "relu activation" {
    var test_mat = try Tensor(f32).fromSlice(.{ 0, 0, 9 }, @constCast(&[_]f32{ 0.0, -1.0, 4.0, 0.0, -1.0, 4.0, 0.0, -1.0, 4.0 }));

    var results = try Tensor(f32).alloc(test_mat.shape, std.testing.allocator);
    defer results.deinit(std.testing.allocator);

    var backprop_results = try Tensor(f32).alloc(test_mat.shape, std.testing.allocator);
    defer backprop_results.deinit(std.testing.allocator);

    ops.relu(f32, &test_mat, &results);
    ops.reluBackprop(f32, &test_mat, &backprop_results);

    try std.testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0 }, results.constSlice());
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 }, backprop_results.constSlice());
}

test "sigmoid activation" {
    var test_mat = try Tensor(f32).allocWithValue(.{ 0, 3, 1 }, 1.0, std.testing.allocator);
    defer test_mat.deinit(std.testing.allocator);

    var sigmoid_mat = try Tensor(f32).alloc(test_mat.shape, std.testing.allocator);
    defer sigmoid_mat.deinit(std.testing.allocator);

    _ = Sigmoid.apply(&test_mat, &sigmoid_mat);
    try std.testing.expectEqual(2.19317573589, ops.sum(f32, &sigmoid_mat));

    test_mat.fill(0.0);
    _ = Sigmoid.apply(&test_mat, &sigmoid_mat);
    try std.testing.expectEqual(1.5, ops.sum(f32, &sigmoid_mat));
}
