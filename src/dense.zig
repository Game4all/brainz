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
        /// Contains the error gradient for this layer.
        /// - Serves as a basis for computing the gradient wrt to the layer weights and biases.
        grad: Matrix(f32),
        /// Contains the computed error gradient to be backpropagated.
        backwards_grad: Matrix(f32),

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

            const w_transposed = self.weights.transpose();
            self.backwards_grad = try Matrix(f32).empty(try root.ops.opResultShape(.Mul, w_transposed.shape, self.grad.shape), alloc);
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

            const w_transposed = self.weights.transpose();
            self.backwards_grad = try Matrix(f32).empty(try root.ops.opResultShape(.Mul, w_transposed.shape, self.grad.shape), alloc);

            @memcpy(self.weights.get_mut_slice(), w);
            @memcpy(self.biases.get_mut_slice(), b);
        }

        /// Performs forward propagation through this layer.
        pub fn forward(self: *@This(), inputs: *Matrix(f32)) *Matrix(f32) {
            // perform linear combination
            root.ops.mul(f32, &self.weights, inputs, &self.last_outputs);
            root.ops.add(f32, &self.biases, &self.last_outputs, &self.last_outputs);

            // apply activation element wise
            return activation.apply(&self.last_outputs, &self.activation_outputs);
        }

        /// Performs backwards propagation for a hidden layer.
        /// - Stores the resulting error gradient inside the `grad` variable.
        /// - Returns the computed err_gradient to pass on for backpropagation.
        pub fn backwards(self: *@This(), err_grad: *const Matrix(f32)) *Matrix(f32) {
            // compute actual error gradient for this layer.
            const activ = activation.apply_derivative(&self.last_outputs, &self.grad);
            root.ops.hadamard(f32, activ, err_grad, &self.grad);

            // compute the gradient that will get passed to the layer before that one for backprop
            const w_transposed = self.weights.transpose();
            root.ops.mul(f32, &w_transposed, &self.grad, &self.backwards_grad);

            return &self.backwards_grad;
        }

        /// Frees the memory of this layer.
        pub fn deinit(self: *@This()) void {
            self.weights.deinit();
            self.biases.deinit();
            self.last_outputs.deinit();
            self.activation_outputs.deinit();
            self.grad.deinit();
            self.backwards_grad.deinit();
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
        @memcpy(inputs.get_mut_slice(), @constCast(&in));
        const prediction = xor_mlp.forward(&inputs);
        try std.testing.expectEqualSlices(f32, &outs, prediction.get_slice());
    }
}

test "linear regression backprop test" {
    const Linear = @import("activation.zig").Linear;
    const MSE = @import("loss.zig").MSE;
    const ops = @import("ops.zig");

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const alloc = arena.allocator();
    const inputs = [_][1]f32{
        [1]f32{0.0},
        [1]f32{1.0},
        [1]f32{3.0},
        [1]f32{9.0},
    };

    // outputs follow f(x)=2x + 1
    const outputs = [_][1]f32{
        [1]f32{1.0},
        [1]f32{3.0},
        [1]f32{7.0},
        [1]f32{19.0},
    };

    var regressor: Dense(1, 1, Linear) = undefined;
    try regressor.init(alloc);

    // contains the expected value for backprop
    var expected_mat = try Matrix(f32).empty(.{ 1, 1 }, alloc);
    // contains the computed loss gradient
    var loss_grad = try Matrix(f32).empty(.{ 1, 1 }, alloc);

    // contains the input of the network
    var input_mat = try Matrix(f32).empty(.{ 1, 1 }, alloc);
    var input_transposed = input_mat.transpose();

    // contains the gradient wrt to the weights
    var weights_grad = try Matrix(f32).empty(.{ 1, 1 }, alloc);

    // train for 100 epochs.
    for (0..100) |_| {
        for (inputs, outputs) |i, o| {
            @memcpy(input_mat.get_mut_slice(), @constCast(&i));
            @memcpy(expected_mat.get_mut_slice(), @constCast(&o));

            const result = regressor.forward(&input_mat);
            MSE.compute_derivative(result, &expected_mat, &loss_grad);

            // compute the gradients for the layer.
            // they are stored in the `.grad`
            _ = regressor.backwards(&loss_grad);
            ops.mul_scalar(f32, &regressor.grad, 0.1, &regressor.grad); // scale the error gradient by 0.1 so we don't have to do it twice for the weight and bias update.

            // compute the grad wrt to the weights.
            ops.mul(f32, &regressor.grad, &input_transposed, &weights_grad);

            // update the weights
            ops.sub(f32, &regressor.weights, &weights_grad, &regressor.weights); // Wnew = Wold - Wgrad;
            // update the bias
            ops.sub(f32, &regressor.biases, &regressor.grad, &regressor.biases); // Bnew = Bold - grad;
        }
    }

    for (inputs, outputs) |i, o| {
        @memcpy(input_mat.get_mut_slice(), @constCast(&i));
        @memcpy(expected_mat.get_mut_slice(), @constCast(&o));

        const result = regressor.forward(&input_mat);
        try std.testing.expectApproxEqAbs(o[0], result.get(.{ 0, 0 }), 1.0e-5);
    }
}
