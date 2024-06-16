const builtin = @import("builtin");
const std = @import("std");
const brainz = @import("brainz");
const Matrix = brainz.Matrix;
const Allocator = std.mem.Allocator;

///
/// Binary classification example with a real dataset
/// Labels: 0 1
///
/// A sample is 48x13 or 624 values
/// 50 first elements have label 0
/// 50 last elements have label 1

// import the dataset in C array format in another file.
const DATASET = @cImport({
    @cInclude("dataset.h");
    @cInclude("eval_data.h");
});

const LABELS: [100]f32 = [_]f32{0.0} ** 50 ++ [_]f32{1.0} ** 50;
const EVAL_LABELS: [10]f32 = [_]f32{0.0} ** 5 ++ [_]f32{1.0} ** 5;

pub fn main() !void {
    if (builtin.mode == .Debug)
        std.log.warn("You're running this example in Debug mode, training may take a really long time. Run in release mode to have the entire example run in a matter of seconds", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    var out = std.io.getStdOut().writer();

    const alloc = arena.allocator();
    var mlp: ClassificationMLP = undefined;
    try mlp.init(alloc);

    var input_mat = try Matrix(f32).empty(.{ 624, 1 }, alloc);
    var expected_mat = try Matrix(f32).empty(.{ 1, 1 }, alloc);
    var loss_grad = try Matrix(f32).empty(.{ 1, 1 }, alloc);

    const BCE = brainz.loss.BinaryCrossEntropy;

    const time_before = try std.time.Instant.now();

    // train the network for 500 epochs.
    for (0..500) |_| {
        for (DATASET.DATASET, LABELS) |i, o| {
            const data: *const [624]f32 = @ptrCast(&i);

            input_mat.setData(@constCast(data));
            expected_mat.set(.{ 0, 0 }, o);

            // forward prop first
            const result = mlp.forward(&input_mat);

            // then backprop through the net layers
            BCE.computeDerivative(result, &expected_mat, &loss_grad);
            mlp.backwards(&loss_grad);

            // then update the network weights
            mlp.step(&input_mat, 0.1);
        }
    }

    const time_after = try std.time.Instant.now();
    try out.print("Took {}ms for training \n", .{time_after.since(time_before) / std.time.ns_per_ms});

    try out.print("========= Evaluating network ==========\n", .{});

    // initialize a RNG for sampling random samples to evaluate.
    var rnd = blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(@constCast(@alignCast(std.mem.asBytes(&seed))));
        var pcg = std.Random.Pcg.init(seed);
        break :blk pcg.random();
    };

    // sample 10 random items from the eval dataset to evaluate model performance.
    for (0..10) |_| {
        const idx = rnd.intRangeAtMost(usize, 0, DATASET.EVAL_DATASET.len - 1);
        const data: *const [624]f32 = @ptrCast(&DATASET.EVAL_DATASET[idx]);
        input_mat.setData(@constCast(data));

        const result = mlp.forward(&input_mat);

        try out.print("Result: {} | Expected: {} \n", .{ @round(result.get(.{ 0, 0 })), EVAL_LABELS[idx] });
    }
}

/// The MLP for binary classification.
const ClassificationMLP = struct {
    layer_1: brainz.Dense(624, 32, brainz.activation.Sigmoid) = undefined,
    layer_2: brainz.Dense(32, 1, brainz.activation.Sigmoid) = undefined,

    // used for storing temporary weights gradients for updating weights.
    weight_grad_1: Matrix(f32),
    weight_grad_2: Matrix(f32),

    pub fn forward(self: *@This(), input: *const Matrix(f32)) *Matrix(f32) {
        const a = self.layer_1.forward(input);
        return self.layer_2.forward(a);
    }

    pub fn backwards(self: *@This(), loss_grad: *const Matrix(f32)) void {
        const A = self.layer_2.backwards(loss_grad);
        _ = self.layer_1.backwards(A);
    }

    /// Update the weights by the specified learning rate
    pub fn step(self: *@This(), ins: *const Matrix(f32), lr: f32) void {
        const layer2_inputs = self.layer_1.activation_outputs.transpose();
        const layer1_inputs = ins.transpose();

        // scale the error gradients as per the learning rate.
        brainz.ops.mul_scalar(f32, &self.layer_1.grad, lr, &self.layer_1.grad);
        brainz.ops.mul_scalar(f32, &self.layer_2.grad, lr, &self.layer_2.grad);

        // compute the actual gradients wrt to the weights of the layers for weight update.
        brainz.ops.mul(f32, &self.layer_1.grad, &layer1_inputs, &self.weight_grad_1);
        brainz.ops.mul(f32, &self.layer_2.grad, &layer2_inputs, &self.weight_grad_2);

        // update the weights.
        brainz.ops.sub(f32, &self.layer_1.weights, &self.weight_grad_1, &self.layer_1.weights);
        brainz.ops.sub(f32, &self.layer_1.biases, &self.layer_1.grad, &self.layer_1.biases);

        // update the biases.
        brainz.ops.sub(f32, &self.layer_2.weights, &self.weight_grad_2, &self.layer_2.weights);
        brainz.ops.sub(f32, &self.layer_2.biases, &self.layer_2.grad, &self.layer_2.biases);
    }

    pub fn init(self: *@This(), alloc: Allocator) !void {
        try self.layer_1.init(alloc);
        try self.layer_2.init(alloc);

        const wg1_shape = try brainz.ops.opShape(
            .Mul,
            self.layer_1.grad.shape,
            .{ 1, 624 },
        );

        const wg2_shape = try brainz.ops.opShape(
            .Mul,
            self.layer_2.grad.shape,
            .{ 1, 32 },
        );

        self.weight_grad_1 = try Matrix(f32).empty(wg1_shape, alloc);
        self.weight_grad_2 = try Matrix(f32).empty(wg2_shape, alloc);
    }
};
