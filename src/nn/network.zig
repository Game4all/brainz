const std = @import("std");
const Allocator = std.mem.Allocator;
const Loss = @import("loss.zig").Loss;

fn LayerInfo(layer: type) struct { usize, usize, []const u8, []const u8 } {
    return .{ @field(layer, "NUM_INPUTS"), @field(layer, "NUM_OUTPUTS"), @field(layer, "LAYER_TYPE"), @field(layer, "LAYER_ACTIVATION") };
}

pub fn Network(structure: []type) type {
    return struct {
        layers: std.meta.Tuple(structure) = undefined,
        last_inputs: []f32 = &[_]f32{},

        /// Initialize the network with random weights and biases
        pub fn init(self: *@This(), alloc: std.mem.Allocator) !void {
            inline for (0..structure.len) |layer_index| {
                try @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).init(alloc);
            }
            self.last_inputs = try alloc.alloc(f32, structure[0].NUM_INPUTS);
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
            self.last_inputs = try alloc.alloc(f32, structure[0].NUM_INPUTS);
        }

        pub fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
            inline for (0..structure.len) |layer_index| {
                @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).deinit(alloc);
            }
            alloc.free(self.last_inputs);
            self.last_inputs = &[_]f32{};
        }

        /// Performs forward propagation through the network.
        pub fn forward(self: *@This(), input: []f32) []f32 {
            std.debug.assert(input.len == self.last_inputs.len);
            @memcpy(self.last_inputs, input);

            var last_outputs: []f32 = &[_]f32{};
            last_outputs = self.layers.@"0".forward(input);

            inline for (1..structure.len) |layer_index|
                last_outputs = @field(self.layers, std.fmt.comptimePrint("{}", .{layer_index})).forward(last_outputs);

            return last_outputs;
        }

        pub fn backwards(self: *@This(), expected_out: []f32, rate: f32, loss: Loss) f32 {
            const num_layers: i32 = @intCast(structure.len - 1);
            comptime var layer_idx = num_layers;

            var total_loss: f32 = 0.0;

            inline while (layer_idx >= 0) : (layer_idx -= 1) {
                if (layer_idx == num_layers) {
                    var last_layer = @field(self.layers, std.fmt.comptimePrint("{}", .{structure.len - 1}));
                    total_loss = last_layer.backwards_out(expected_out, loss);
                } else {
                    var current_layer = @field(self.layers, std.fmt.comptimePrint("{}", .{layer_idx}));
                    const previous_layer = @field(self.layers, std.fmt.comptimePrint("{}", .{layer_idx + 1}));
                    current_layer.backwards(previous_layer.gamma, previous_layer.weights, structure[@intCast(layer_idx + 1)].NUM_OUTPUTS);
                }
            }

            inline for (0..structure.len) |layer| {
                if (layer == 0) {
                    self.layers.@"0".update_weights(self.last_inputs, rate);
                } else {
                    const prev_outputs = @field(self.layers, std.fmt.comptimePrint("{}", .{layer - 1})).last_outputs;
                    @field(self.layers, std.fmt.comptimePrint("{}", .{layer})).update_weights(prev_outputs, rate);
                }
            }

            return total_loss;
        }

        /// Prints info about the network to the console.
        pub fn network_info(self: *@This()) void {
            std.log.info("=============Network structure=============", .{});
            inline for (structure, 0..) |layer, layer_idx| {
                const in, const outs, const str, const activation = LayerInfo(layer);
                std.log.info("Layer #{} : Type={s} | Inputs={} | Outputs={} | Activation={s}", .{ layer_idx, str, in, outs, activation });
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
    const Heaviside = @import("activation.zig").Heaviside;

    const XorMLP = Network(@constCast(&[_]type{
        DenseLayer(2, 2, Heaviside),
        DenseLayer(2, 1, Heaviside),
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
        try std.testing.expectEqualSlices(f32, &outs, prediction);
    }
}

test "multi layer perceptron XOR backpropagation learning test" {
    const DenseLayer = @import("dense.zig").DenseLayer;
    const Sigmoid = @import("activation.zig").Sigmoid;
    const MSE = @import("loss.zig").MSE;

    const XorMlp = Network(@constCast(&[_]type{
        DenseLayer(2, 2, Sigmoid),
        DenseLayer(2, 1, Sigmoid),
    }));

    var net: XorMlp = .{};
    try net.init(std.testing.allocator);
    defer net.deinit(std.testing.allocator);

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

    for (0..10_000) |_| {
        for (inputs, outputs) |ins, outs| {
            _ = net.forward(@constCast(&ins));
            _ = net.backwards(@constCast(&outs), 0.5, MSE);
        }
    }

    for (inputs, outputs) |ins, outs| {
        const prediction = net.forward(@constCast(&ins));
        try std.testing.expectApproxEqAbs(outs[0], prediction[0], 0.1);
    }
}
