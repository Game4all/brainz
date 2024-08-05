const builtin = @import("builtin");
const std = @import("std");
const brainz = @import("brainz");
const Tensor = brainz.Tensor;
const Allocator = std.mem.Allocator;
const Device = brainz.Device;

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

    var device: brainz.default_device = .{};
    try device.init(arena.allocator(), 3);
    defer device.deinit();

    var out = std.io.getStdOut().writer();

    const alloc = arena.allocator();
    var mlp: ClassificationMLP = undefined;
    try mlp.init(alloc);

    var loss_grad = try Tensor(f32).init(mlp.outputShape(), alloc);

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
            const input_mat = try Tensor(f32).initFromSlice(mlp.inputShape(), @constCast(data));
            const expected_mat = try Tensor(f32).initFromSlice(mlp.outputShape(), @constCast(LABELS[i..(i + 10)]));

            const result = try mlp.forward(device.device(), &input_mat);

            loss = brainz.ops.binaryCrossEntropyLoss(f32, device.device(), result, &expected_mat);
            try brainz.ops.binaryCrossEntropyLossBackprop(f32, device.device(), result, &expected_mat, &loss_grad);
            try device.device().barrier();

            try mlp.backwards(device.device(), &loss_grad);
            try mlp.step(device.device(), &input_mat, 0.2);
        }

        const epoch_end_time = try std.time.Instant.now();
        const elapsed_time = epoch_end_time.since(epoch_start_time) / std.time.ns_per_ms;
        const num_iter_per_s = std.time.ms_per_s / elapsed_time;

        try out.print("\r [{} / {}] loss: {} ({} ms/epoch | {} epochs/s)                        ", .{ a, num_epochs, loss, elapsed_time, num_iter_per_s });
    }

    const time_after = try std.time.Instant.now();
    try out.print("\nTook {}ms for training \n", .{time_after.since(time_before) / std.time.ns_per_ms});

    try out.print("========= Evaluating network ==========\n", .{});

    const eval_inputs = try Tensor(f32).initFromSlice(.{ 10, 624, 1 }, @as(*[6240]f32, @constCast(@ptrCast(&DATASET.EVAL_DATASET[0])))[0..]);
    const eval_labels = try Tensor(f32).initFromSlice(.{ 10, 1, 1 }, @constCast(@ptrCast(EVAL_LABELS[0..])));

    const results = try mlp.forward(device.device(), &eval_inputs);

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

    pub fn forward(self: *@This(), device: Device, input: *const Tensor(f32)) !*Tensor(f32) {
        const a = try self.layer_1.forward(device, input);
        const b = try self.layer_2.forward(device, a);
        return b;
    }

    pub fn backwards(self: *@This(), device: Device, loss_grad: *const Tensor(f32)) !void {
        const A = try self.layer_2.backwards(device, loss_grad);
        _ = try self.layer_1.backwards(device, A);
    }

    pub inline fn inputShape(self: *@This()) struct { usize, usize, usize } {
        return self.layer_1.inputShape();
    }

    pub inline fn outputShape(self: *@This()) struct { usize, usize, usize } {
        return self.layer_2.outputShape();
    }

    /// Update the weights by the specified learning rate
    pub fn step(self: *@This(), device: Device, ins: *const Tensor(f32), lr: f32) !void {
        const layer2_inputs = self.layer_1.activation_outputs.transpose();
        const layer1_inputs = ins.transpose();

        // compute the batched gradients wrt to the layers weights
        try brainz.ops.matMul(f32, device, &self.layer_1.grad, &layer1_inputs, &self.weight_grad_1);
        try brainz.ops.matMul(f32, device, &self.layer_2.grad, &layer2_inputs, &self.weight_grad_2);

        try device.barrier();

        // sum the batched gradients for the weight and bias gradients
        try brainz.ops.reduce(f32, device, .Sum, &self.weight_grad_1, 0, &self.weight_grad_1_f);
        try brainz.ops.reduce(f32, device, .Sum, &self.weight_grad_2, 0, &self.weight_grad_2_f);
        try brainz.ops.reduce(f32, device, .Sum, &self.layer_1.grad, 0, &self.bias_grad_1);
        try brainz.ops.reduce(f32, device, .Sum, &self.layer_2.grad, 0, &self.bias_grad_2);

        try device.barrier();

        // update the weights and biases for each layer scaling the gradients by the batch size and learning rate
        try brainz.ops.sub(f32, device, &self.layer_1.weights, &self.weight_grad_1_f, &self.layer_1.weights, .{ .alpha = lr * 0.1 });
        try brainz.ops.sub(f32, device, &self.layer_1.biases, &self.bias_grad_1, &self.layer_1.biases, .{ .alpha = lr * 0.1 });

        try brainz.ops.sub(f32, device, &self.layer_2.weights, &self.weight_grad_2_f, &self.layer_2.weights, .{ .alpha = lr * 0.1 });
        try brainz.ops.sub(f32, device, &self.layer_2.biases, &self.bias_grad_2, &self.layer_2.biases, .{ .alpha = lr * 0.1 });

        try device.barrier();
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

        self.weight_grad_1 = try Tensor(f32).init(wg1_shape, alloc);
        self.weight_grad_2 = try Tensor(f32).init(wg2_shape, alloc);

        self.weight_grad_1_f = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, wg1_shape, 0), alloc);
        self.weight_grad_2_f = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, wg2_shape, 0), alloc);

        self.bias_grad_1 = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.layer_1.outputShape(), 0), alloc);
        self.bias_grad_2 = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.layer_2.outputShape(), 0), alloc);
    }
};
