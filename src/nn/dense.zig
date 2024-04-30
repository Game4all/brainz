const math = @import("../math.zig");
const Df32 = math.Df32;

const std = @import("std");
const Activation = @import("activation.zig").Activation;
const Allocator = std.mem.Allocator;

/// A layer of densely connected neurons.
/// TODO: Use a different approach to compute derivatives instead of using a dual number representation for differentiation.
pub fn DenseLayer(comptime num_in: usize, comptime num_out: usize, comptime activation: Activation) type {
    return struct {
        /// Layer weights.
        weights: []f32 = &[_]f32{},
        /// Layer biases.
        biases: []f32 = &[_]f32{},
        /// The last computed output values of this layer.
        last_outputs: []f32 = &[_]f32{},
        ///------------------------------------- Backward propagation stuff ----------------------
        gamma: []f32 = &[_]f32{},

        pub const NUM_INPUTS = num_in;
        pub const NUM_OUTPUTS = num_out;
        pub const LAYER_TYPE = "DENSE";

        /// Initialize the layer with random weights and bias values for the specified number of inputs.
        pub fn init(self: *@This(), allocator: Allocator) !void {
            const variance = 2.0 / @as(f32, @floatFromInt((num_in + num_out)));

            const weights = try allocator.alloc(f32, num_out * num_in);
            errdefer allocator.free(weights);

            const biases = try allocator.alloc(f32, num_out);
            errdefer allocator.free(biases);

            var rnd = blk: {
                var seed: u64 = undefined;
                try std.posix.getrandom(@constCast(@alignCast(std.mem.asBytes(&seed))));
                var pcg = std.Random.Pcg.init(seed);
                break :blk pcg.random();
            };

            // iniialise randomly weights and biases
            for (weights) |*weight|
                weight.* = rnd.floatNorm(f32) * variance;

            for (biases) |*bias|
                bias.* = rnd.floatNorm(f32) * variance;

            try self.init_weights(weights, biases, allocator);
        }

        /// Initialize the layer with the given `weights` and `biases`.
        /// NOTE:  The layer becomes the owner of the weights and biases memory.
        pub fn init_weights(self: *@This(), weights: []f32, biases: []f32, allocator: Allocator) !void {
            std.debug.assert(weights.len == num_in * num_out);
            std.debug.assert(biases.len == num_out);

            self.biases = biases;
            self.weights = weights;
            self.last_outputs = try allocator.alloc(f32, num_out);
            self.gamma = try allocator.alloc(f32, num_out);
        }

        /// Performs forward propagation of the inputs through this perceptron layer.
        pub fn forward(self: *@This(), inputs: []f32) []f32 {
            std.debug.assert(inputs.len == num_in);

            for (0..num_out, self.biases) |n_idx, bias| {
                var weighted_val: f32 = 0.0;

                for (inputs, 0..) |value, w_idx| {
                    weighted_val += value * self.weights[n_idx * num_in + w_idx];
                }

                weighted_val += bias;

                const val = activation.apply(weighted_val, self.last_outputs);
                self.last_outputs[n_idx] = val;
            }

            return self.last_outputs;
        }

        /// Performs backward propagation of the loss values for the output layer.
        pub fn backwards_out(self: *@This(), expected_outputs: []f32) void {
            for (self.gamma, self.last_outputs, expected_outputs) |*gamma, last_out, expected_output| {
                gamma.* = (expected_output - last_out) * activation.apply_derivative(last_out, self.last_outputs);
            }
        }

        /// Performs backward propagation of the loss values.
        pub fn backwards(self: *@This(), prev_gamma: []f32, prev_weights: []f32, prev_num_out: usize) void {
            for (self.gamma, self.last_outputs, 0..) |*gamma, last_activ, input_idx| {
                var summed: f32 = 0.0;
                for (0..prev_num_out) |out_idx|
                    summed += prev_gamma[out_idx] * prev_weights[out_idx * self.last_outputs.len + input_idx];

                gamma.* = activation.apply_derivative(last_activ, self.last_outputs) * summed;
            }
        }

        pub fn update_weights(self: *@This(), inputs: []f32, rate: f32) void {
            for (self.biases, self.gamma, 0..) |*bias, loss, n_idx| {
                const w_grad = loss * rate;
                for (0..num_in) |weight_index|
                    self.weights[n_idx * num_in + weight_index] = self.weights[n_idx * num_in + weight_index] + w_grad * inputs[weight_index];

                bias.* += w_grad;
            }
        }

        pub fn deinit(self: *@This(), allocator: Allocator) void {
            allocator.free(self.weights);
            allocator.free(self.biases);
            allocator.free(self.last_outputs);
            // backprop
            allocator.free(self.gamma);
        }
    };
}