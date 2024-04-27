const std = @import("std");
const Allocator = std.mem.Allocator;

const BackpropParams = @import("backprop.zig").BackpropParams;

fn LayerInfo(layer: type) struct { usize, usize, []const u8 } {
    return .{ @field(layer, "NUM_INPUTS"), @field(layer, "NUM_OUTPUTS"), @field(layer, "LAYER_TYPE") };
}

pub fn Network(structure: []type) type {
    return struct {
        layers: std.meta.Tuple(structure) = undefined,

        /// Initialize the network with random weights and biases
        pub fn init(self: *@This(), alloc: std.mem.Allocator) !void {
            inline for (0..structure.len) |layer_index| {
                try @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).init(alloc);
            }
        }

        /// Initialize the each layer of the network with the specified weights and biases.
        pub fn init_weights(self: *@This(), weights: [][]f32, biases: [][]f32, alloc: std.mem.Allocator) !void {
            inline for (0..structure.len) |layer_index| {
                const weights_copy = try alloc.alloc(f32, weights[layer_index].len);
                const biases_copy = try alloc.alloc(f32, biases[layer_index].len);

                @memcpy(weights_copy, weights[layer_index]);
                @memcpy(biases_copy, biases[layer_index]);

                try @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).init_weights(weights_copy, biases_copy, alloc);
            }
        }

        pub fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
            inline for (0..structure.len) |layer_index| {
                @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).deinit(alloc);
            }
        }

        /// Performs forward propagation through the network.
        pub fn forward(self: *@This(), input: []f32) []f32 {
            var last_outputs: []f32 = &[_]f32{};
            last_outputs = self.layers.@"0".forward(input);

            inline for (1..structure.len) |layer_index|
                last_outputs = @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).forward(last_outputs);

            return last_outputs;
        }

        pub fn backprop(self: *@This(), desired_out: []f32, input: []f32, params: BackpropParams, alloc: std.mem.Allocator) !f32 {

            // allocating the error gradients for backpropagation
            const error_grads = try alloc.alloc([]f32, structure.len);
            defer alloc.free(error_grads);
            inline for (error_grads, 0..structure.len) |*grad, layer_index| {
                grad.* = try alloc.alloc(f32, structure[layer_index].NUM_OUTPUTS);
                defer alloc.free(grad.*);
            }

            const num_layers: i32 = @intCast(structure.len - 1);

            // calculating loss gradients for previous layers
            comptime var layer_idx = num_layers;
            inline while (layer_idx >= 0) : (layer_idx -= 1) {
                if (layer_idx == num_layers) { // calculate the loss for the last layer
                    const last_layer = @field(self.layers, std.fmt.comptimePrint("{}", .{structure.len - 1}));
                    for (error_grads[structure.len - 1], desired_out, last_layer.last_outputs) |*grad, desired_output, last_output|
                        grad.* = desired_output - last_output;
                } else {
                    const current_layer = @field(self.layers, std.fmt.comptimePrint("{}", .{layer_idx}));
                    const previous_layer = @field(self.layers, std.fmt.comptimePrint("{}", .{layer_idx + 1}));

                    for (error_grads[layer_idx], current_layer.last_activ_outputs, 0..) |*grad, last_activ, input_idx| {
                        var summed: f32 = 0.0;
                        for (0..previous_layer.last_activ_outputs.len) |n_idx| {
                            const r = error_grads[layer_idx + 1][n_idx] * previous_layer.weights[n_idx * current_layer.last_activ_outputs.len + input_idx];
                            summed += r;
                        }
                        grad.* = last_activ.value * last_activ.derivative * summed;
                    }
                }
            }

            var total_e: f32 = 0.0;

            total_e += self.layers.@"0".backprop(input, error_grads[0], params);

            inline for (1..structure.len) |layer_index| {
                // std.log.info("Updating weights of layer {}", .{layer_index});
                const previous_layer = @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index - 1}));
                total_e += @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).backprop(previous_layer.last_outputs, error_grads[layer_index], params);
            }

            return total_e;
        }

        /// Prints info about the network to the console.
        pub fn network_info(self: *@This()) void {
            std.log.info("=============Network structure=============", .{});
            inline for (structure, 0..) |layer, layer_idx| {
                const in, const outs, const str = LayerInfo(layer);
                std.log.info("Layer #{} : Type={s} | Inputs={} | Outputs={}", .{ layer_idx, str, in, outs });
            }

            std.log.info("==============Network weights==============", .{});
            inline for (0..structure.len) |layer_index| {
                const weights = @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).weights;
                const biases = @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).biases;
                std.log.info("Layer #{} Weights: {any}", .{ layer_index, weights });
                std.log.info("Layer #{} Biases: {any}", .{ layer_index, biases });
            }
        }
    };
}

test "multi layer perceptron XOR test" {
    const DenseLayer = @import("dense.zig").DenseLayer;
    const unit_step = @import("../math.zig").unit_step;

    const XorMLP = Network(@constCast(&[_]type{
        DenseLayer(2, 2, unit_step),
        DenseLayer(2, 1, unit_step),
    }));

    const weights = [_][]f32{
        @constCast(&[_]f32{ 1.0, 1.0, 1.0, 1.0 }),
        @constCast(&[_]f32{ 1.0, -1.0 }),
    };

    const biases = [_][]f32{
        @constCast(&[_]f32{ -0.5, -1.5 }),
        @constCast(&[_]f32{0.0}),
    };

    var network: XorMLP = .{};
    try network.init_weights(@constCast(&weights), @constCast(&biases), std.testing.allocator);
    defer network.deinit(std.testing.allocator);

    const inputs = [_][2]f32{
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

    for (inputs, outputs) |ins, outs| {
        const prediction = network.forward(@constCast(&ins));
        try std.testing.expect(std.mem.eql(f32, prediction, &outs));
    }
}
