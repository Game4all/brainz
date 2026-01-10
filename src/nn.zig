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
        pub fn initializeWeights(self: *const Self, rnd: std.Random) void {
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

const ActivationFunc = enum {
    sigmoid,
    relu,
};

/// A wrapper layer over an activation function (ReLU, Sigmoid) op.
/// # Args
/// - `activ`: The activation function to apply (relu, sigmoid) to the inputs of this layers.
/// # Returns
/// Output tensor of same shape as the input.
pub fn Activation(comptime activ: ActivationFunc) type {
    return struct {
        const Self = @This();

        /// Returns an initialized layer
        pub const init: Self = .{};

        pub fn forward(self: *const @This(), plan: *LinearPlan, input: *const Tensor) !*const Tensor {
            _ = self;
            return switch (activ) {
                .relu => try ops.relu(plan, input),
                .sigmoid => try ops.sigmoid(plan, input),
            };
        }
    };
}

/// Validates that a type follows the Layer API at comptime.
/// A layer MUST have the following function signatures implemented:
/// - `fn forward(*const Self, *LinearPlan, *const Tensor) !*const Tensor`
fn assertIsLayer(comptime L: type) void {

    // check if the layer even has a forward() method
    if (!std.meta.hasMethod(L, "forward"))
        @compileError(std.fmt.comptimePrint("Layer type {s} doesn't have a forward method.", .{@typeName(L)}));

    const fnArgs = std.meta.ArgsTuple(@TypeOf(L.forward));
    const fields = std.meta.fields(fnArgs);

    if (fields.len != 3)
        @compileError(std.fmt.comptimePrint("Layer type {}'s `forward` method doesn't have the correct signature. Expected three arguments: (*const self, *LinearPlan, *const Tensor) ", .{@typeName(L)}));

    const planTy = fields[fields.len - 2].type;
    const tensorTy = fields[fields.len - 1].type;

    if (planTy != *LinearPlan or tensorTy != *const Tensor)
        @compileError(std.fmt.comptimePrint("Layer type {}'s `forward` method doesn't have the correct signature. Expected three arguments: (*const self, *LinearPlan, *const Tensor) ", .{@typeName(L)}));
}

/// A comptime wrapper for neural nets that execute in a sequence
/// This provides automatically helpers for the forward pass based on a struct that describes the net architecture.
/// # Note
/// The architecture struct must implement an init function to initialize the network layers.
pub fn Sequential(comptime T: type) type {

    // assert that we passed in a struct and that all struct fields follow the layer API.
    comptime switch (@typeInfo(T)) {
        .@"struct" => |s| {
            for (s.fields) |field|
                assertIsLayer(field.type);
        },
        else => @compileError("Network must be initialized with a struct type"),
    };

    // assert the struct also has an init function to initialize all layers
    if (!@hasDecl(T, "init"))
        @compileError("The architecture struct must have an 'init' function to initialize the layers");

    return struct {
        const Self = @This();

        layers: T,

        /// Initializes the network layers.
        /// # Args
        /// - `args`: a tuple/struct passed to the inner struct's init function.
        pub fn init(args: anytype) !Self {
            return .{
                .layers = try @call(.auto, T.init, args),
            };
        }

        /// Performs a forward pass through all layers in the declared order.
        /// # Args
        /// - `input` is the tensor to be used for forward pass of this net.
        pub fn forward(self: *const Self, plan: *LinearPlan, input: *const Tensor) !*const Tensor {
            var current_input = input;

            inline for (std.meta.fields(T)) |field| {
                current_input = try @field(self.layers, field.name).forward(plan, current_input);
            }

            return current_input;
        }

        /// Initializes the weights of the layers holding parameters to sensible defaults using the provided random source.
        pub fn initializeWeights(self: *const Self, rng: std.Random) void {
            inline for (std.meta.fields(T)) |field| {
                if (std.meta.hasMethod(field.type, "initializeWeights")) {
                    @field(self.layers, field.name).initializeWeights(rng);
                }
            }
        }

        /// Returns the underlying architecture struct for manual access to the layers.
        pub fn getInner(self: *const Self) *const T {
            return &self.layers;
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

test "Sequential: testing automatic forward pass" {
    var memArena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var planBuilder: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer planBuilder.deinit();

    // declare a test architecture with two linear layers and a reLu inbetween.
    const TestNet = struct {
        l1: Linear(f32, true),
        act: Activation(.relu),
        l2: Linear(f32, true),

        pub fn init(plan: *LinearPlan) !@This() {
            return .{
                .l1 = try Linear(f32, true).init(plan, 3, 4),
                .act = .{},
                .l2 = try Linear(f32, true).init(plan, 4, 2),
            };
        }
    };

    const Net = Sequential(TestNet);
    const net = try Net.init(.{&planBuilder});

    // create a dummy input tensor
    const input = try planBuilder.createParam(.float32, Shape.fromSlice(&.{ 1, 3 }));
    // get the output to compare its shape to the expected dimension shapes
    const output = try net.forward(&planBuilder, input);

    // expected output dims is (1, 2)
    try std.testing.expectEqual(2, output.shape.n_dimensions);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, output.shape.dimensions[0..output.shape.n_dimensions]);
}
