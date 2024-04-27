const math = @import("../math.zig");
const Df32 = math.Df32;

const std = @import("std");
const Allocator = std.mem.Allocator;

const BackpropParams = @import("backprop.zig").BackpropParams;

/// A layer of densely connected neurons.
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
        /// The last inputs of this layer.
        last_inputs: []f32 = &[_]f32{},

        pub const NUM_INPUTS = num_in;
        pub const NUM_OUTPUTS = num_out;
        pub const LAYER_TYPE = "DENSE";

        /// Initialize the layer with random weights and bias values for the specified number of inputs.
        pub fn init(self: *@This(), allocator: Allocator) !void {
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
                weight.* = rnd.floatNorm(f32);

            for (biases) |*bias|
                bias.* = rnd.floatNorm(f32);

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
            self.last_inputs = try allocator.alloc(f32, num_in);
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

            @memcpy(self.last_inputs, inputs);

            return self.last_outputs;
        }

        /// Performs backpropagation based on the results of a previous feed forward.
        /// Parameters:
        /// `input` -> The inputs
        /// `loss` -> The loss values computed for each output of this network.
        pub fn backprop(self: *@This(), input: []f32, loss: []f32, params: BackpropParams) f32 {
            var total_err: f32 = 0.0;

            for (0..num_out, self.biases, loss) |out_idx, *bias, delta| {
                const w_grad = std.math.clamp(delta * params.learn_rate, -params.grad_clip_norm, params.grad_clip_norm);
                for (0..num_in) |weight_index| {
                    self.weights[out_idx * num_in + weight_index] += self.weights[out_idx * num_in + weight_index] * w_grad * input[weight_index];
                }

                bias.* += (bias.*) * w_grad;
                total_err += @abs(loss[out_idx]);
            }

            return total_err;
        }

        pub fn deinit(self: *@This(), allocator: Allocator) void {
            allocator.free(self.weights);
            allocator.free(self.biases);
            allocator.free(self.last_outputs);
            allocator.free(self.last_activ_outputs);
            allocator.free(self.last_inputs);
        }
    };
}
