const std = @import("std");
const root = @import("root.zig");

const Activation = @import("activation.zig").Activation;
const Allocator = std.mem.Allocator;
const Matrix = @import("matrix.zig").Matrix;

/// A layer of densely connected neurons.
pub fn Dense(comptime num_in: usize, comptime num_out: usize, comptime activation: Activation) type {
    return struct {
        /// The layer's weights.
        weights: Matrix(f32),
        /// The layer's biases
        biases: Matrix(f32),
        /// The last computed linear combinations of this layer.
        last_outputs: Matrix(f32),
        /// The last activation outputs of this layer.
        activation_outputs: Matrix(f32),
        /// The gradients of this layer
        grad: Matrix(f32),

        /// Initializes the layer with the given allocator.
        /// The weights and biases are randomly initialized.
        pub fn init(self: *@This(), alloc: Allocator) !void {
            const rnd = blk: {
                var seed: u64 = undefined;
                try std.posix.getrandom(@constCast(@alignCast(std.mem.asBytes(&seed))));
                var pcg = std.Random.Pcg.init(seed);
                break :blk pcg.random();
            };

            self.weights = try Matrix(f32).random(.{ num_out, num_in }, rnd, alloc);
            self.biases = try Matrix(f32).random(.{ num_out, 1 }, rnd, alloc);
            self.last_outputs = try Matrix(f32).empty(try root.ops.opResultShape(.Mul, self.weights.shape, .{ num_in, 1 }), alloc);
            self.activation_outputs = try Matrix(f32).empty(self.last_outputs.shape, alloc);
            self.grad = try Matrix(f32).empty(self.last_outputs.shape, alloc);
        }

        /// Initializes the layer with the given allocator and given weights.
        pub fn init_with_weights(self: *@This(), alloc: Allocator, w: []f32, b: []f32) !void {
            std.debug.assert(w.len == num_in * num_out);
            std.debug.assert(b.len == num_out);

            self.weights = try Matrix(f32).empty(.{ num_out, num_in }, alloc);
            self.biases = try Matrix(f32).empty(.{ num_out, 1 }, alloc);
            self.last_outputs = try Matrix(f32).empty(try root.ops.opResultShape(.Mul, self.weights.shape, .{ num_in, 1 }), alloc);
            self.activation_outputs = try Matrix(f32).empty(self.last_outputs.shape, alloc);
            self.grad = try Matrix(f32).empty(self.last_outputs.shape, alloc);

            @memcpy(self.weights.to_slice(), w);
            @memcpy(self.biases.to_slice(), b);
        }

        /// Performs forward propagation through this layer.
        pub fn forward(self: *@This(), inputs: *Matrix(f32)) *Matrix(f32) {
            // perform linear combination
            root.ops.mul(f32, &self.weights, inputs, &self.last_outputs);
            root.ops.add(f32, &self.biases, &self.last_outputs, &self.last_outputs);

            // apply activation element wise
            return activation.apply(&self.last_outputs, &self.activation_outputs);
        }

        /// Performs backwards propagation for the output layer.
        /// Stores the resulting gradient inside the `grad` variable.
        pub fn backwards_output(
            self: *@This(),
            loss_grad: *Matrix(f32),
        ) *Matrix(f32) {
            const activ = activation.apply_derivative(&self.last_outputs, &self.grad);
            root.ops.hadamard(f32, activ, loss_grad, &self.grad);
            return &self.grad;
        }

        // pub fn backwards(self: *@This(), loss_grad: *Matrix(f32), weights: *Matrix(f32)) *Matrix(f32) {}

        /// Frees the memory of this layer.
        pub fn deinit(self: *@This()) void {
            self.weights.deinit();
            self.biases.deinit();
            self.last_outputs.deinit();
            self.activation_outputs.deinit();
            self.grad.deinit();
        }
    };
}

test "basic xor mlp" {
    const activations = @import("activation.zig");

    const XorMLP = struct {
        layer_1: Dense(2, 2, activations.Heaviside) = undefined,
        layer_2: Dense(2, 1, activations.Heaviside) = undefined,

        pub fn forward(self: *@This(), input: *Matrix(f32)) *Matrix(f32) {
            const a = self.layer_1.forward(input);
            return self.layer_2.forward(a);
        }

        pub fn init(self: *@This(), alloc: Allocator) !void {
            try self.layer_1.init_with_weights(
                alloc,
                @constCast(&[_]f32{ 1.0, 1.0, 1.0, 1.0 }),
                @constCast(&[_]f32{ -0.5, -1.5 }),
            );
            try self.layer_2.init_with_weights(
                alloc,
                @constCast(&[_]f32{ 1.0, -1.0 }),
                @constCast(&[_]f32{0.0}),
            );
        }

        pub fn deinit(self: *@This()) void {
            self.layer_1.deinit();
            self.layer_2.deinit();
        }
    };

    var xor_mlp: XorMLP = undefined;
    try xor_mlp.init(std.testing.allocator);
    defer xor_mlp.deinit();

    var inputs = try Matrix(f32).empty(.{ 2, 1 }, std.testing.allocator);
    defer inputs.deinit();

    const ins = [_][2]f32{
        [2]f32{ 0.0, 0.0 },
        [2]f32{ 0.0, 1.0 },
        [2]f32{ 1.0, 0.0 },
        [2]f32{ 1.0, 1.0 },
    };

    const outputs = [_][1]f32{
        [1]f32{0.0},
        [1]f32{1.0},
        [1]f32{1.0},
        [1]f32{0.0},
    };

    for (ins, outputs) |in, outs| {
        @memcpy(inputs.to_slice(), @constCast(&in));
        const prediction = xor_mlp.forward(&inputs);
        try std.testing.expectEqualSlices(f32, &outs, prediction.to_slice());
    }
}
