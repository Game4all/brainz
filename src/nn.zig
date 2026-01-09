const std = @import("std");
const nn = @import("nn.zig");
const tensor = @import("tensor.zig");
const prog = @import("plan.zig");
const ops = @import("ops.zig");

const Tensor = tensor.Tensor;
const TensorArena = tensor.TensorArena;
const Shape = tensor.Shape;
const LinearPlan = prog.LinearPlan;
const Dtype = tensor.Dtype;

/// A linear (=fully connected) layer for a neural network.
///
/// This layer basically does the the y = xw + (b) if bias is enabled. Atcivations are to be manually specified.
///
/// # Args
/// - `ty`: The data type for the layer parameters (e.g., f32, f64)
/// - `bias`: Whether to include a bias to the layer.
///
/// # Expected Shapes
/// - Input: `(batch_size, in_features)`
/// - Weights: `(in_features, out_features)`
/// - Biases: `(out_features)` (if bias was enabled)
/// - Output: `(batch_size, out_features)`
pub fn Linear(comptime ty: type, comptime bias: bool) type {
    return struct {
        const Self = @This();

        /// Weight matrix of shape (in_features, out_features)
        weights: *const Tensor,
        /// Optional bias vector of shape (out_features)
        biases: ?*const Tensor,
        /// Number of input features
        in_features: usize,
        /// Number of output features
        out_features: usize,

        /// Initializes a linear layer with the specified dimensions.
        ///
        /// # Parameters
        /// - `plan`: The linear plan to register parameters with
        /// - `in_features`: Number of input features
        /// - `out_features`: Number of output features
        ///
        /// # Returns
        /// A new Linear layer instance with uninitialized parameters.
        pub fn init(plan: *LinearPlan, in_features: usize, out_features: usize) !Self {
            const dtype = comptime Dtype.getBackingDType(ty);

            const weights = try plan.createParam(dtype, Shape.fromSlice(&.{ in_features, out_features }));
            const biases = if (bias) try plan.createParam(dtype, Shape.fromSlice(&.{out_features})) else null;

            return .{
                .weights = weights,
                .biases = biases,
                .in_features = in_features,
                .out_features = out_features,
            };
        }

        /// Initializes the weights of the layer using the provided random generator.
        pub fn randomizeWeights(self: *const Self, rnd: std.Random) void {
            if (self.weights.slice(f32)) |storage| {
                for (storage) |*val|
                    val.* = rnd.floatNorm(f32) * 0.1;
            }

            if (self.biases) |b| {
                if (b.slice(f32)) |storage| {
                    for (storage) |*val|
                        val.* = rnd.floatNorm(f32) * 0.1;
                }
            }
        }

        /// Performs a forward pass of the layer.
        ///
        /// # Args
        /// - `plan`: The linear plan to append operations to
        /// - `input`: Input tensor of shape `(batch_size, in_features)`.
        ///
        ///
        /// # Returns
        /// Output tensor of shape `(batch_size, out_features)`
        pub fn forward(self: *const Self, plan: *LinearPlan, input: *const Tensor) !*const Tensor {
            const xw = try ops.matmul(plan, input, self.weights);
            return if (self.biases) |b| try ops.add(plan, xw, b) else xw;
        }
    };
}

test "Linear layer: initialization and shape" {
    var memArena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var planBuilder: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer planBuilder.deinit();

    // layer with 3 -> 5 features including bias
    const LinearLayer = nn.Linear(f32, true);
    const layer = try LinearLayer.init(&planBuilder, 3, 5);

    // dims should be 3 input features, 5 out features
    try std.testing.expectEqual(3, layer.in_features);
    try std.testing.expectEqual(5, layer.out_features);

    // weights should be (3, 5)
    try std.testing.expectEqual(2, layer.weights.shape.n_dimensions);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 5 }, layer.weights.shape.dimensions[0..layer.weights.shape.n_dimensions]);

    // bias shape should be (5) basically
    try std.testing.expect(layer.biases != null);
    if (layer.biases) |b| {
        try std.testing.expectEqual(1, b.shape.n_dimensions);
        try std.testing.expectEqualSlices(usize, &[_]usize{5}, b.shape.dimensions[0..b.shape.n_dimensions]);
    }
}
