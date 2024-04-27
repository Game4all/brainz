const math = @import("../math.zig");
const Df32 = math.Df32;

const std = @import("std");
const Allocator = std.mem.Allocator;

const BackpropParams = @import("backprop.zig").BackpropParams;

/// A layer of densely connected neurons.
/// TODO: Allow for an inference_only mode.
pub fn DenseLayer(comptime num_in: usize, comptime num_out: usize, comptime activation: fn (x: anytype) Df32) type {
    return struct {
        /// Layer weights.
        weights: []f32 = &[_]f32{},
        /// Layer biases.
        biases: []f32 = &[_]f32{},
        /// The last computed output values of this layer.
        last_outputs: []f32 = &[_]f32{},
        /// The last computed output values of this layer with their derivatives.
        /// Used for backwards propagation.
        last_activ_outputs: []Df32 = &[_]Df32{},

        ///------------------------------------- Backward propagation stuff ----------------------
        gamma: []f32 = &[_]f32{},
        inputs: []f32 = &[_]f32{},

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
                try std.os.getrandom(std.mem.asBytes(&seed));
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
            self.last_activ_outputs = try allocator.alloc(Df32, num_out);

            self.gamma = try allocator.alloc(f32, num_out);
            self.inputs = try allocator.alloc(f32, num_in);
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

                const val = activation(Df32.with_grad(weighted_val));
                self.last_activ_outputs[n_idx] = val;
                self.last_outputs[n_idx] = val.value;
            }

            @memcpy(self.inputs, inputs);

            return self.last_outputs;
        }

        /// Performs backward propagation of the loss values for the output layer.
        pub fn backwards_out(self: *@This(), expected_outputs: []f32) void {
            for (self.gamma, self.last_activ_outputs, expected_outputs) |*gamma, last_out, expected_output| {
                gamma.* = (expected_output - last_out.value) * last_out.derivative;
            }
        }

        pub fn backwards(self: *@This(), prev_gamma: []f32, prev_weights: []f32, prev_num_out: usize) void {
            for (self.gamma, self.last_activ_outputs, 0..) |*gamma, last_activ, input_idx| {
                var summed: f32 = 0.0;
                for (0..prev_num_out) |out_idx|
                    summed += prev_gamma[out_idx] * prev_weights[out_idx * self.last_activ_outputs.len + input_idx];

                gamma.* = last_activ.derivative * summed;
            }
        }

        pub fn update_weights(self: *@This(), rate: f32) void {
            for (self.biases, self.gamma, 0..) |*bias, loss, n_idx| {
                const w_grad = loss * rate;
                for (0..num_in) |weight_index|
                    self.weights[n_idx * num_in + weight_index] = self.weights[n_idx * num_in + weight_index] + w_grad * self.inputs[weight_index];

                bias.* += w_grad;
            }
        }

        pub fn deinit(self: *@This(), allocator: Allocator) void {
            allocator.free(self.weights);
            allocator.free(self.biases);
            allocator.free(self.last_outputs);
            allocator.free(self.last_activ_outputs);

            // backprop
            allocator.free(self.gamma);
            allocator.free(self.inputs);
        }
    };
}
