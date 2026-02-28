const std = @import("std");
const nn = @import("nn.zig");
const tensor = @import("tensor.zig");
const prog = @import("plan.zig");
const ops = @import("ops.zig");

const Tensor = tensor.Tensor;
const TensorArena = tensor.TensorArena;
const Shape = tensor.Shape;
const LinearPlan = prog.LinearPlan;
const PlanBuilder = prog.PlanBuilder;
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
    const param_count = if (bias) 2 else 1;

    return struct {
        const Self = @This();

        /// Weight matrix of shape (in_features, out_features) + bias matrix of shape (out_features) if enabled
        params: [param_count]*const Tensor,
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
        pub fn init(plan: *PlanBuilder, in_features: usize, out_features: usize) !Self {
            const dtype = comptime Dtype.getBackingDType(ty);

            const params = blk: {
                const weights = try plan.createTensor(dtype, Shape.fromSlice(&.{ in_features, out_features }), true);
                if (bias) {
                    const biases = try plan.createTensor(dtype, Shape.fromSlice(&.{out_features}), true);
                    break :blk [_]*const Tensor{ weights, biases };
                }
                break :blk [_]*const Tensor{weights};
            };

            return .{
                .params = params,
                .in_features = in_features,
                .out_features = out_features,
            };
        }

        /// Initializes the weights of the layer using the provided random generator.
        pub fn initializeWeights(self: *const Self, rnd: std.Random) void {
            for (self.params) |p| {
                if (p.slice(f32)) |storage| {
                    for (storage) |*val|
                        val.* = rnd.floatNorm(ty) * 0.1;
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
        pub fn forward(self: *const Self, plan: *PlanBuilder, input: *const Tensor) !*const Tensor {
            const xw = try ops.matmul(plan, input, self.params[0]);
            return if (bias) try ops.add(plan, xw, self.params[1]);
        }

        /// Returns the trainable parameters of this layer
        pub fn parameters(self: *const Self) []const *const Tensor {
            return &self.params;
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

        pub fn forward(self: *const @This(), plan: *PlanBuilder, input: *const Tensor) !*const Tensor {
            _ = self;
            return switch (activ) {
                .relu => try ops.relu(plan, input),
                .sigmoid => try ops.sigmoid(plan, input),
            };
        }

        /// Returns the trainable parameters of this layer (which there is none of since it's an activation function without any trainable parameters)
        pub fn parameters(_: *const Self) []const *const Tensor {
            return &.{};
        }
    };
}

fn checkHasMethod(comptime L: type, comptime name: [:0]const u8, comptime expectedArgs: []const type) void {
    if (!std.meta.hasMethod(L, name))
        @compileError(std.fmt.comptimePrint("Layer type {s} doesn't have a '{s}' method which is part of the Layer API.", .{ @typeName(L), name }));

    const fnArgs = std.meta.ArgsTuple(@TypeOf(@field(L, name)));
    const argsList = std.meta.fields(fnArgs);
    if (expectedArgs.len != argsList.len)
        @compileError(std.fmt.comptimePrint("Layer type {s} has a mismatched argument count on '{s}' method (expected {d} args, got {d})", .{ @typeName(L), name, expectedArgs.len, argsList.len }));

    inline for (argsList, 0..) |fld, i| {
        if (expectedArgs[i] != fld.type)
            @compileError(std.fmt.comptimePrint("Layer type {s} has a mismatched argument on '{s}' method (expected arg #{d} to be of type {s}, got {s} instead)", .{ @typeName(L), name, i, @typeName(expectedArgs[i]), @typeName(fld.type) }));
    }
}

/// Validates that a type follows the Layer API at comptime.
/// A layer MUST have the following function signatures implemented:
/// - An init function or field returning a built layer.
/// - A forward pass method: `fn forward(*const Self, *PlanBuilder, *const Tensor) !*const Tensor`
/// - A `parameters` method which returns trainable parameters in the layer: `fn parameters(*const Self) []const *const Tensor`
fn assertIsLayer(comptime L: type) void {
    // check if the layer has an init function or init field
    if (!@hasDecl(L, "init") and !@hasField(L, "init"))
        @compileError(std.fmt.comptimePrint("Layer type {s} doesn't have an init function or init field.", .{@typeName(L)}));

    // check if the layer has a forward function
    checkHasMethod(L, "forward", &.{ *const L, *PlanBuilder, *const Tensor });
    // check if the layer has a parameters function
    checkHasMethod(L, "parameters", &.{*const L});
}

/// A comptime wrapper for neural nets that execute in a sequence
/// This provides automatically helpers for the forward pass based on a struct that describes the network architecture.
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
        /// - `args`: a tuple/struct passed to the inner architecture struct's init function.
        pub fn init(args: anytype) !Self {
            return .{
                .layers = try @call(.auto, T.init, args),
            };
        }

        /// Performs a forward pass through all layers in the declared order.
        /// # Args
        /// - `input` is the tensor to be used for forward pass of this net.
        pub fn forward(self: *const Self, plan: *PlanBuilder, input: *const Tensor) !*const Tensor {
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

        /// Returns the total number of trainable parameters in the network.
        pub fn getParameterCount(self: *const Self) usize {
            var paramCount: usize = 0;
            inline for (std.meta.fields(T)) |fld| {
                const params = @call(.auto, @field(fld.type, "parameters"), .{&@field(self.layers, fld.name)});
                paramCount += params.len;
            }
            return paramCount;
        }

        /// Returns an allocated slice of all trainable parameters in the network.
        pub fn collectParameters(self: *const Self, allocator: std.mem.Allocator) ![]const *const Tensor {
            const slice = try allocator.alloc(*const Tensor, self.getParameterCount());
            var i: usize = 0;
            inline for (std.meta.fields(T)) |fld| {
                const params: []const *const Tensor = @field(self.layers, fld.name).parameters();
                @memcpy(slice[i..(i + params.len)], params);
                i += params.len;
            }
            return slice;
        }

        /// Saves tensor weights to the specified stream for the current model instance.
        /// NOTE: This method only serializes tensor weights and does not include any information about the computational graph structure, which is defined by the code itself.
        pub fn saveWeights(self: *const Self, writer: *std.Io.Writer) !void {
            // write total parameter tensor count before tensor data
            try writer.writeInt(usize, self.getParameterCount(), .little);

            inline for (std.meta.fields(T)) |fld| {
                const params: []const *const Tensor = @field(self.layers, fld.name).parameters();
                if (params.len != 0) {
                    for (params) |param|
                        try param.writeContents(writer);
                }
            }
        }

        /// Loads previously serialized tensor weights and biases from a stream
        /// and restores them into the current model instance.
        ///
        /// NOTE: This method only deserializes tensor data and does not validate that
        /// the computational graph structure is the same as it was for the serialized model,
        /// as the graph is defined by code. Ensure that the model architecture matches
        /// the serialization source before calling this method.
        pub fn loadWeights(self: *const Self, reader: *std.Io.Reader) !void {
            // read total parameter tensor count to ensure architecture is very roughly the same as the serialized files
            const totalParamCount = try reader.peekInt(usize, .little);
            reader.toss(@sizeOf(usize));

            if (totalParamCount != self.getParameterCount())
                return error.MismatchedParameterCount;

            inline for (std.meta.fields(T)) |fld| {
                const params: []const *const Tensor = @field(self.layers, fld.name).parameters();
                if (params.len != 0) {
                    for (params) |param|
                        try param.readContents(reader);
                }
            }
        }
    };
}

test "Linear layer: initialization and shape" {
    var memArena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer memArena.deinit();

    var planBuilder: LinearPlan = .init(memArena.allocator());
    defer planBuilder.deinit();

    const builder = &planBuilder.builder;

    // layer with 3 -> 5 features including bias
    const layer: nn.Linear(f32, true) = try .init(builder, 3, 5);

    // we should have two parameters, one for weights and the other for biases
    try std.testing.expectEqual(2, layer.params.len);

    const weights = layer.params[0];

    // dims should be 3 input features, 5 out features
    try std.testing.expectEqual(3, layer.in_features);
    try std.testing.expectEqual(5, layer.out_features);

    // weights should be (3, 5)
    try std.testing.expectEqual(2, weights.shape.n_dimensions);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 5 }, weights.shape.dimensions[0..weights.shape.n_dimensions]);

    const bias = layer.params[1];

    // bias shape should be (5) basically
    try std.testing.expectEqual(1, bias.shape.n_dimensions);
    try std.testing.expectEqualSlices(usize, &[_]usize{5}, bias.shape.dimensions[0..bias.shape.n_dimensions]);

    // check total parameter count
    const totalParams = layer.parameters();
    try std.testing.expectEqual(2, totalParams.len);
}

test "Sequential: testing automatic forward pass" {
    var memArena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer memArena.deinit();

    var linearPlan: LinearPlan = .init(memArena.allocator());
    defer linearPlan.deinit();

    const builder = &linearPlan.builder;

    // declare a test architecture with two linear layers and a reLu inbetween.
    const TestNet = struct {
        l1: Linear(f32, true),
        act: Activation(.relu),
        l2: Linear(f32, true),

        pub fn init(plan: *PlanBuilder) !@This() {
            return .{
                .l1 = try .init(plan, 3, 4),
                .act = .init,
                .l2 = try .init(plan, 4, 2),
            };
        }
    };

    const net: Sequential(TestNet) = try .init(.{builder});

    // create a dummy input tensor
    const input = try builder.createTensor(.float32, Shape.fromSlice(&.{ 1, 3 }), true);
    // get the output to compare its shape to the expected dimension shapes
    const output = try net.forward(builder, input);

    // expected output dims is (1, 2)
    try std.testing.expectEqual(2, output.shape.n_dimensions);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, output.shape.dimensions[0..output.shape.n_dimensions]);

    // check total parameter count (should be 4, since we have 2 layers with weights and biases tensors)
    const totalParams = net.getParameterCount();
    try std.testing.expectEqual(4, totalParams);
}

test "Sequential: testing model serialization" {
    var memArena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer memArena.deinit();

    var planBuilder: LinearPlan = .init(memArena.allocator());
    defer planBuilder.deinit();
    const builder = &planBuilder.builder;

    const net: Sequential(struct {
        l1: Linear(f32, true),
        act: Activation(.relu),
        l2: Linear(f32, true),

        pub fn init(plan: *PlanBuilder) !@This() {
            return .{
                .l1 = try .init(plan, 3, 4),
                .act = .init,
                .l2 = try .init(plan, 4, 2),
            };
        }
    }) = try .init(.{builder});

    const input = try builder.createTensor(.float32, Shape.fromSlice(&.{ 1, 3 }), true);
    _ = try net.forward(builder, input);

    const executablePlan = try planBuilder.finalize(true);
    _ = executablePlan;

    var testBuffer = std.Io.Writer.Allocating.init(memArena.allocator());
    const writer = &testBuffer.writer;

    try net.saveWeights(writer);

    var reader = std.Io.Reader.fixed(testBuffer.written());
    try net.loadWeights(&reader);
}
