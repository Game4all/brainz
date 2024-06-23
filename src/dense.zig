const std = @import("std");
const root = @import("root.zig");

const Activation = @import("activation.zig").Activation;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

/// A layer of densely connected neurons.
pub fn Dense(comptime num_in: usize, comptime num_out: usize, comptime batch_size: usize, comptime activation: Activation) type {
    return struct {
        /// The layer's weights.
        weights: Tensor(f32),
        /// The layer's biases
        biases: Tensor(f32),
        /// The last computed linear combinations of this layer.
        last_outputs: Tensor(f32),
        /// The last activation outputs of this layer.
        activation_outputs: Tensor(f32),
        /// Contains the error gradient for this layer.
        /// - Serves as a basis for computing the gradient wrt to the layer weights and biases.
        grad: Tensor(f32),
        /// Contains the computed error gradient to be backpropagated.
        backwards_grad: Tensor(f32),

        /// Initializes the layer with the given allocator.
        /// The weights and biases are randomly initialized.
        pub fn init(self: *@This(), alloc: Allocator) !void {
            const rnd = blk: {
                var seed: u64 = undefined;
                try std.posix.getrandom(@constCast(@alignCast(std.mem.asBytes(&seed))));
                var pcg = std.Random.Pcg.init(seed);
                break :blk pcg.random();
            };

            self.weights = try Tensor(f32).random(.{ 0, num_out, num_in }, rnd, alloc);
            self.biases = try Tensor(f32).random(.{ 0, num_out, 1 }, rnd, alloc);
            self.last_outputs = try Tensor(f32).empty(try root.ops.opShape(.MatMul, self.weights.shape, .{ batch_size, num_in, 1 }), alloc);
            self.activation_outputs = try Tensor(f32).empty(self.last_outputs.shape, alloc);
            self.grad = try Tensor(f32).empty(self.last_outputs.shape, alloc);

            const w_transposed = self.weights.transpose();
            self.backwards_grad = try Tensor(f32).empty(try root.ops.opShape(.MatMul, w_transposed.shape, self.grad.shape), alloc);
        }

        /// Initializes the layer with the given allocator and given weights.
        pub fn initWithWeights(self: *@This(), alloc: Allocator, w: []f32, b: []f32) !void {
            std.debug.assert(w.len == num_in * num_out);
            std.debug.assert(b.len == num_out);

            self.weights = try Tensor(f32).empty(.{ 0, num_out, num_in }, alloc);
            self.biases = try Tensor(f32).empty(.{ 0, num_out, 1 }, alloc);

            self.last_outputs = try Tensor(f32).empty(try root.ops.opShape(.MatMul, self.weights.shape, self.inputShape()), alloc);
            self.activation_outputs = try Tensor(f32).empty(self.last_outputs.shape, alloc);

            self.grad = try Tensor(f32).empty(self.last_outputs.shape, alloc);

            const w_transposed = self.weights.transpose();
            self.backwards_grad = try Tensor(f32).empty(try root.ops.opShape(.MatMul, w_transposed.shape, self.grad.shape), alloc);

            @memcpy(self.weights.slice(), w);
            @memcpy(self.biases.slice(), b);
        }

        /// Performs forward propagation through this layer.
        pub fn forward(self: *@This(), inputs: *const Tensor(f32)) *Tensor(f32) {
            // perform linear combination
            root.ops.matMul(f32, &self.weights, inputs, &self.last_outputs);
            root.ops.add(f32, &self.biases, &self.last_outputs, &self.last_outputs);

            // apply activation element wise
            return activation.apply(&self.last_outputs, &self.activation_outputs);
        }

        /// Performs backwards propagation for a hidden layer.
        /// - Stores the resulting error gradient inside the `grad` variable.
        /// - Returns the computed err_gradient to pass on for backpropagation.
        pub fn backwards(self: *@This(), err_grad: *const Tensor(f32)) *Tensor(f32) {
            // compute actual error gradient for this layer.
            const activ = activation.applyDerivative(&self.last_outputs, &self.grad);
            root.ops.mul(f32, activ, err_grad, &self.grad);

            // compute the gradient that will get passed to the layer before that one for backprop
            const w_transposed = self.weights.transpose();
            root.ops.matMul(f32, &w_transposed, &self.grad, &self.backwards_grad);

            return &self.backwards_grad;
        }

        pub inline fn inputShape(_: *@This()) struct { usize, usize, usize } {
            return .{ batch_size, num_in, 1 };
        }

        pub inline fn outputShape(_: *@This()) struct { usize, usize, usize } {
            return .{ batch_size, num_out, 1 };
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

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const alloc = arena.allocator();

    const XorMLP = struct {
        layer_1: Dense(2, 2, 0, activations.Heaviside) = undefined,
        layer_2: Dense(2, 1, 0, activations.Heaviside) = undefined,

        pub fn forward(self: *@This(), input: *Tensor(f32)) *Tensor(f32) {
            const a = self.layer_1.forward(input);
            return self.layer_2.forward(a);
        }

        pub fn init(self: *@This(), allocator: Allocator) !void {
            try self.layer_1.initWithWeights(
                allocator,
                @constCast(&[_]f32{ 1.0, 1.0, 1.0, 1.0 }),
                @constCast(&[_]f32{ -0.5, -1.5 }),
            );
            try self.layer_2.initWithWeights(
                allocator,
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
    try xor_mlp.init(alloc);

    var inputs = try Tensor(f32).empty(.{ 0, 2, 1 }, alloc);

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
        @memcpy(inputs.slice(), @constCast(&in));
        const prediction = xor_mlp.forward(&inputs);
        try std.testing.expectEqualSlices(f32, &outs, prediction.constSlice());
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

    var regressor: Dense(1, 1, 0, Linear) = undefined;
    try regressor.init(alloc);

    // contains the expected value for backprop
    var expected_mat = try Tensor(f32).empty(.{ 0, 1, 1 }, alloc);
    // contains the computed loss gradient
    var loss_grad = try Tensor(f32).empty(.{ 0, 1, 1 }, alloc);

    // contains the input of the network
    var input_mat = try Tensor(f32).empty(.{ 0, 1, 1 }, alloc);
    var input_transposed = input_mat.transpose();

    // contains the gradient wrt to the weights
    var weights_grad = try Tensor(f32).empty(.{ 0, 1, 1 }, alloc);

    // train for 100 epochs.
    for (0..100) |_| {
        for (inputs, outputs) |i, o| {
            @memcpy(input_mat.slice(), @constCast(&i));
            @memcpy(expected_mat.slice(), @constCast(&o));

            const result = regressor.forward(&input_mat);
            MSE.computeDerivative(result, &expected_mat, &loss_grad);

            // compute the gradients for the layer.
            // they are stored in the `.grad`
            _ = regressor.backwards(&loss_grad);
            ops.mulScalar(f32, &regressor.grad, 0.1, &regressor.grad); // scale the error gradient by 0.1 so we don't have to do it twice for the weight and bias update.

            // compute the grad wrt to the weights.
            ops.matMul(f32, &regressor.grad, &input_transposed, &weights_grad);

            // update the weights
            ops.sub(f32, &regressor.weights, &weights_grad, &regressor.weights); // Wnew = Wold - Wgrad;
            // update the bias
            ops.sub(f32, &regressor.biases, &regressor.grad, &regressor.biases); // Bnew = Bold - grad;
        }
    }

    for (inputs, outputs) |i, o| {
        @memcpy(input_mat.slice(), @constCast(&i));
        @memcpy(expected_mat.slice(), @constCast(&o));

        const result = regressor.forward(&input_mat);
        try std.testing.expectApproxEqAbs(o[0], result.get(.{ 0, 0, 0 }), 1.0e-5);
    }
}
