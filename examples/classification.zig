const builtin = @import("builtin");
const std = @import("std");
const brainz = @import("brainz");
const Tensor = brainz.Matrix;
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

    var input_mat = try Tensor(f32).alloc(mlp.inputShape(), alloc);
    var expected_mat = try Tensor(f32).alloc(mlp.outputShape(), alloc);
    var loss_grad = try Tensor(f32).alloc(mlp.outputShape(), alloc);

    const BCE = brainz.loss.BinaryCrossEntropy;

    const time_before = try std.time.Instant.now();

    // train the network for 500 epochs.
    // do minibatch training with a batch_size of 10.
    const num_epochs = 1000;
    for (0..num_epochs) |a| {
        var i: usize = 0;
        var loss: f32 = 0.0;
        const epoch_start_time = try std.time.Instant.now();

        while (i < 100) : (i += 10) {
            const data: *const [6240]f32 = @ptrCast(&DATASET.DATASET[i]);
            input_mat.setData(data);
            expected_mat.setData(@constCast(LABELS[i..(i + 10)]));

            const result = mlp.forward(&input_mat);
            BCE.computeDerivative(result, &expected_mat, &loss_grad);
            loss = BCE.compute(result, &expected_mat);

            mlp.backwards(&loss_grad);
            mlp.step(&input_mat, 0.20);
        }

        const epoch_end_time = try std.time.Instant.now();
        const elapsed_time = epoch_end_time.since(epoch_start_time) / std.time.ns_per_ms;
        const num_iter_per_s = std.time.ms_per_s / elapsed_time;

        try out.print("\r [{} / {}] loss: {} ({} ms/epoch | {} epochs/s)                        ", .{ a, num_epochs, loss, elapsed_time, num_iter_per_s });
    }

    const time_after = try std.time.Instant.now();
    try out.print("\nTook {}ms for training \n", .{time_after.since(time_before) / std.time.ns_per_ms});

    try out.print("========= Evaluating network ==========\n", .{});

    const eval_inputs = try Tensor(f32).fromSlice(.{ 10, 624, 1 }, @as(*[6240]f32, @constCast(@ptrCast(&DATASET.EVAL_DATASET[0])))[0..]);
    const eval_labels = try Tensor(f32).fromSlice(.{ 10, 1, 1 }, @constCast(@ptrCast(EVAL_LABELS[0..])));

    const results = mlp.forward(&eval_inputs);

    for (0..10) |i| {
        try out.print("Out: {} | Expected: {} \n", .{ @round(results.get(.{ 0, i, 0 })), eval_labels.get(.{ i, 0, 0 }) });
    }
}

/// The MLP for binary classification.
const ClassificationMLP = struct {
    layer_1: brainz.Dense(624, 32, 10, brainz.activation.Sigmoid) = undefined,
    layer_2: brainz.Dense(32, 1, 10, brainz.activation.Sigmoid) = undefined,

    // used for storing temporary weights gradients for updating weights.
    weight_grad_1: Tensor(f32),
    weight_grad_2: Tensor(f32),

    // flattened and averaged weight gradients
    weight_grad_1_f: Tensor(f32),
    weight_grad_2_f: Tensor(f32),

    bias_grad_1: Tensor(f32),
    bias_grad_2: Tensor(f32),

    pub fn forward(self: *@This(), input: *const Tensor(f32)) *Tensor(f32) {
        const a = self.layer_1.forward(input);
        return self.layer_2.forward(a);
    }

    pub fn backwards(self: *@This(), loss_grad: *const Tensor(f32)) void {
        const A = self.layer_2.backwards(loss_grad);
        _ = self.layer_1.backwards(A);
    }

    pub inline fn inputShape(self: *@This()) struct { usize, usize, usize } {
        return self.layer_1.inputShape();
    }

    pub inline fn outputShape(self: *@This()) struct { usize, usize, usize } {
        return self.layer_2.outputShape();
    }

    /// Update the weights by the specified learning rate
    pub fn step(self: *@This(), ins: *const Tensor(f32), lr: f32) void {
        const layer2_inputs = self.layer_1.activation_outputs.transpose();
        const layer1_inputs = ins.transpose();

        // compute the batched gradients wrt to the layers weights
        brainz.ops.matMul(f32, &self.layer_1.grad, &layer1_inputs, &self.weight_grad_1);
        brainz.ops.matMul(f32, &self.layer_2.grad, &layer2_inputs, &self.weight_grad_2);

        // sum the batched gradients for the weight and bias gradients
        brainz.ops.reduce(f32, .Sum, &self.weight_grad_1, 0, &self.weight_grad_1_f);
        brainz.ops.reduce(f32, .Sum, &self.weight_grad_2, 0, &self.weight_grad_2_f);
        brainz.ops.reduce(f32, .Sum, &self.layer_1.grad, 0, &self.bias_grad_1);
        brainz.ops.reduce(f32, .Sum, &self.layer_2.grad, 0, &self.bias_grad_2);

        // average the batched weight gradients and scale them wrt learning rate
        brainz.ops.mulScalar(f32, &self.weight_grad_1_f, lr * 0.1, &self.weight_grad_1_f);
        brainz.ops.mulScalar(f32, &self.weight_grad_2_f, lr * 0.1, &self.weight_grad_2_f);

        // average the batched bias gradients and scale them wrt learning rate
        brainz.ops.mulScalar(f32, &self.bias_grad_1, lr * 0.1, &self.bias_grad_1);
        brainz.ops.mulScalar(f32, &self.bias_grad_2, lr * 0.1, &self.bias_grad_2);

        // update the weights and biases for each layer
        brainz.ops.sub(f32, &self.layer_1.weights, &self.weight_grad_1_f, &self.layer_1.weights);
        brainz.ops.sub(f32, &self.layer_1.biases, &self.bias_grad_1, &self.layer_1.biases);

        brainz.ops.sub(f32, &self.layer_2.weights, &self.weight_grad_2_f, &self.layer_2.weights);
        brainz.ops.sub(f32, &self.layer_2.biases, &self.bias_grad_2, &self.layer_2.biases);
    }

    pub fn init(self: *@This(), alloc: Allocator) !void {
        try self.layer_1.init(alloc);
        try self.layer_2.init(alloc);

        const wg1_shape = try brainz.ops.opShape(
            .MatMul,
            self.layer_1.grad.shape,
            try brainz.ops.opShape(.Transpose, self.layer_1.inputShape(), null),
        );

        const wg2_shape = try brainz.ops.opShape(
            .MatMul,
            self.layer_2.grad.shape,
            try brainz.ops.opShape(.Transpose, self.layer_1.outputShape(), null),
        );

        self.weight_grad_1 = try Tensor(f32).alloc(wg1_shape, alloc);
        self.weight_grad_2 = try Tensor(f32).alloc(wg2_shape, alloc);

        self.weight_grad_1_f = try Tensor(f32).alloc(try brainz.ops.opShape(.SumAxis, wg1_shape, 0), alloc);
        self.weight_grad_2_f = try Tensor(f32).alloc(try brainz.ops.opShape(.SumAxis, wg2_shape, 0), alloc);

        self.bias_grad_1 = try Tensor(f32).alloc(try brainz.ops.opShape(.SumAxis, self.layer_1.outputShape(), 0), alloc);
        self.bias_grad_2 = try Tensor(f32).alloc(try brainz.ops.opShape(.SumAxis, self.layer_2.outputShape(), 0), alloc);
    }
};
