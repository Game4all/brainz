const std = @import("std");
const math = @import("math.zig");
const Df32 = math.Df32;

/// A perceptron with a number `num_inputs` of inputs and the `activation` activation function.
/// Makes use of zig SIMD vectors for vectorization.
pub fn Perceptron(comptime num_inputs: usize, comptime activation: fn (x: anytype) Df32) type {
    return struct {
        pub const Input = @Vector(num_inputs, f32);
        pub const Weights = @Vector(num_inputs + 1, f32);

        /// The weights of the inputs + the bias.
        weights: Weights,

        pub fn init(initial_weights: ?[num_inputs + 1]f32) !@This() {
            if (initial_weights) |i_weights| {
                return .{ .weights = i_weights };
            } else {
                var rnd = blk: {
                    var seed: u64 = undefined;
                    try std.posix.getrandom(@constCast(@alignCast(std.mem.asBytes(&seed))));
                    var pcg = std.Random.Pcg.init(seed);
                    break :blk pcg.random();
                };

                var rnd_weights = std.mem.zeroes([num_inputs + 1]f32);
                for (&rnd_weights) |*value|
                    value.* = rnd.floatNorm(f32);

                return .{ .weights = rnd_weights };
            }
        }

        /// Propagates the given inputs through this perceptron.
        /// Returns the value predicted by this perceptron.
        pub fn forward(self: *const @This(), inputs: Input) f32 {
            const extended_input = std.simd.join(inputs, @as(@Vector(1, f32), @splat(1.0)));
            const z = Df32.literal(@reduce(.Add, @as(Weights, self.weights) * extended_input));
            return activation(z).value;
        }

        // Performs backwards propagation to tune the weights and bias.
        // Returns the calculated error.
        pub fn backwards(self: *@This(), inputs: Input, d: f32, a: f32) f32 {
            const extended_input = std.simd.join(inputs, @as(@Vector(1, f32), @splat(1.0)));

            // compute the the result of the forward propagation through the activation func with its derivative.
            const activ_output = activation(Df32.with_grad(@reduce(.Add, @as(Weights, self.weights) * extended_input)));
            const err = d - activ_output.value;

            const adj_rate = a * err * activ_output.derivative;
            var updated_weights = @as(Weights, self.weights);
            updated_weights += @as(Weights, @splat(adj_rate)) * extended_input;

            self.weights = updated_weights;

            return @abs(err);
        }
    };
}

test "basic OR perceptron" {
    const testing = std.testing;

    var perceptron = try Perceptron(2, math.unit_step).init([_]f32{ 1.0, 1.0, -0.5 });
    try testing.expect(perceptron.forward(.{ 0.0, 0.0 }) == 0.0);
    try testing.expect(perceptron.forward(.{ 1.0, 0.0 }) == 1.0);
    try testing.expect(perceptron.forward(.{ 1.0, 1.0 }) == 1.0);
    try testing.expect(perceptron.forward(.{ 0.0, 1.0 }) == 1.0);
}

test "basic AND" {
    const testing = std.testing;
    var perceptron = try Perceptron(2, math.unit_step).init([_]f32{ 1.0, 1.0, -1.0 });
    try testing.expect(perceptron.forward(.{ 0.0, 0.0 }) == 0.0);
    try testing.expect(perceptron.forward(.{ 1.0, 0.0 }) == 0.0);
    try testing.expect(perceptron.forward(.{ 1.0, 1.0 }) == 1.0);
    try testing.expect(perceptron.forward(.{ 0.0, 1.0 }) == 0.0);
}

test "linear regression" {
    const testing = std.testing;
    var perceptron = try Perceptron(1, math.linear).init(null);

    for (0..150_000) |_| {
        // points for lin reg y = 2x + 1
        _ = perceptron.backwards(.{0.0}, 1.0, 0.0002);
        _ = perceptron.backwards(.{1.0}, 3.0, 0.0002);
        _ = perceptron.backwards(.{2.0}, 5.0, 0.0002);
        _ = perceptron.backwards(.{5.0}, 11.0, 0.0002);
    }

    const eval = perceptron.forward(.{1.0});
    try testing.expectApproxEqAbs(3.0, eval, 0.1);
}
