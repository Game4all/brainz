const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Device = @import("device/Device.zig");
const ops = @import("ops.zig");

/// Represents an activation function.
pub const Activation = struct {
    /// Applies the activation function to the input.
    apply: *const fn (Device, *const Tensor(f32), *Tensor(f32)) anyerror!*Tensor(f32),
    ///Applies the derivative of the activation function w.r.t the inputs.
    applyDerivative: *const fn (Device, *const Tensor(f32), *Tensor(f32)) anyerror!*Tensor(f32),
    /// Name of this activation function. Used for display purposes.
    name: [:0]const u8,
};

/// Linear activation function
pub const Linear: Activation = .{
    .apply = linear_activation,
    .applyDerivative = linear_derivative,
    .name = @tagName(.linear),
};

fn linear_activation(_: Device, in: *const Tensor(f32), _: *Tensor(f32)) !*Tensor(f32) {
    return @constCast(in);
}

fn linear_derivative(_: Device, _: *const Tensor(f32), out: *Tensor(f32)) !*Tensor(f32) {
    out.fill(1.0);
    return out;
}

/// REctified Linear Unit (ReLu) activation function
pub const ReLu: Activation = .{
    .apply = relu_activation,
    .applyDerivative = relu_derivative,
    .name = @tagName(.relu),
};

fn relu_activation(device: Device, in: *const Tensor(f32), out: *Tensor(f32)) !*Tensor(f32) {
    try ops.relu(f32, device, in, out);
    try device.barrier();
    return out;
}

fn relu_derivative(device: Device, in: *const Tensor(f32), out: *Tensor(f32)) !*Tensor(f32) {
    try ops.reluBackprop(f32, device, in, out);
    try device.barrier();
    return out;
}

/// Sigmoid activation function
pub const Sigmoid: Activation = .{
    .apply = sigmoid_activation,
    .applyDerivative = sigmoid_derivative,
    .name = @tagName(.sigmoid),
};

fn sigmoid_activation(device: Device, in: *const Tensor(f32), out: *Tensor(f32)) !*Tensor(f32) {
    try ops.sigmoid(f32, device, in, out);
    return out;
}

fn sigmoid_derivative(device: Device, in: *const Tensor(f32), out: *Tensor(f32)) !*Tensor(f32) {
    try ops.sigmoidBackprop(f32, device, in, out);
    return out;
}

/// Sigmoid linear unit activation function
pub const SiLu: Activation = .{
    .apply = silu_activation,
    .applyDerivative = silu_derivative,
    .name = @tagName(.silu),
};

fn silu_activation(dev: Device, in: *const Tensor(f32), out: *Tensor(f32)) !*Tensor(f32) {
    try ops.silu(f32, dev, in, out);
    return out;
}

fn silu_derivative(dev: Device, in: *const Tensor(f32), out: *Tensor(f32)) !*Tensor(f32) {
    try ops.siluBackprop(f32, dev, in, out);
    return out;
}

/// Heaviside aka step unit activation function
pub const Heaviside: Activation = .{
    .apply = heaviside_activation,
    .applyDerivative = heaviside_derivative,
    .name = @tagName(.heaviside),
};

fn heaviside_activation(_: Device, in: *const Tensor(f32), out: *Tensor(f32)) !*Tensor(f32) {
    for (in.constSlice(), out.slice()) |i, *o| {
        o.* = @max(std.math.sign(i), 0);
    }
    return out;
}

fn heaviside_derivative(_: Device, _: *const Tensor(f32), out: *Tensor(f32)) !*Tensor(f32) {
    out.fill(1.0);
    return out;
}

test "relu activation" {
    var test_mat = try Tensor(f32).initFromSlice(.{ 0, 0, 9 }, @constCast(&[_]f32{ 0.0, -1.0, 4.0, 0.0, -1.0, 4.0, 0.0, -1.0, 4.0 }));

    var results = try Tensor(f32).init(test_mat.shape, std.testing.allocator);
    defer results.deinit(std.testing.allocator);

    var backprop_results = try Tensor(f32).init(test_mat.shape, std.testing.allocator);
    defer backprop_results.deinit(std.testing.allocator);

    try ops.relu(f32, Device.DummyDevice, &test_mat, &results);
    try ops.reluBackprop(f32, Device.DummyDevice, &test_mat, &backprop_results);

    try std.testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0 }, results.constSlice());
    try std.testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 }, backprop_results.constSlice());
}

test "sigmoid activation" {
    var test_mat = try Tensor(f32).init(.{ 0, 3, 1 }, std.testing.allocator);
    test_mat.fill(1.0);
    defer test_mat.deinit(std.testing.allocator);

    var sigmoid_mat = try Tensor(f32).init(test_mat.shape, std.testing.allocator);
    defer sigmoid_mat.deinit(std.testing.allocator);

    _ = try Sigmoid.apply(Device.DummyDevice, &test_mat, &sigmoid_mat);
    try std.testing.expectEqual(2.19317573589, ops.sum(f32, &sigmoid_mat));

    test_mat.fill(0.0);
    _ = try Sigmoid.apply(Device.DummyDevice, &test_mat, &sigmoid_mat);
    try std.testing.expectEqual(1.5, ops.sum(f32, &sigmoid_mat));
}
