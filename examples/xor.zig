const std = @import("std");
const brainz = @import("brainz");
const Tensor = brainz.Tensor;
const Allocator = std.mem.Allocator;
const Device = brainz.Device;

const XorMLP = struct {
    layer_1: brainz.Dense(2, 2, 4, brainz.activation.Sigmoid) = undefined,
    layer_2: brainz.Dense(2, 1, 4, brainz.activation.Sigmoid) = undefined,

    weight_grad_1: Tensor(f32),
    weight_grad_2: Tensor(f32),

    // flattened and averaged weight gradients
    weight_grad_1_f: Tensor(f32),
    weight_grad_2_f: Tensor(f32),

    bias_grad_1: Tensor(f32),
    bias_grad_2: Tensor(f32),

    pub fn forward(self: *@This(), device: Device, input: *const Tensor(f32)) !*Tensor(f32) {
        const a = try self.layer_1.forward(device, input);
        return try self.layer_2.forward(device, a);
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

        // sum the batched gradients for the weight and bias gradients
        try brainz.ops.reduce(f32, device, .Sum, &self.weight_grad_1, 0, &self.weight_grad_1_f);
        try brainz.ops.reduce(f32, device, .Sum, &self.weight_grad_2, 0, &self.weight_grad_2_f);
        try brainz.ops.reduce(f32, device, .Sum, &self.layer_1.grad, 0, &self.bias_grad_1);
        try brainz.ops.reduce(f32, device, .Sum, &self.layer_2.grad, 0, &self.bias_grad_2);

        // update the weights and biases for each layer, scaling the gradients by the batch size and learning rate
        try brainz.ops.sub(f32, device, &self.layer_1.weights, &self.weight_grad_1_f, &self.layer_1.weights, .{ .alpha = lr * 0.25 });
        try brainz.ops.sub(f32, device, &self.layer_1.biases, &self.bias_grad_1, &self.layer_1.biases, .{ .alpha = lr * 0.25 });

        try brainz.ops.sub(f32, device, &self.layer_2.weights, &self.weight_grad_2_f, &self.layer_2.weights, .{ .alpha = lr * 0.25 });
        try brainz.ops.sub(f32, device, &self.layer_2.biases, &self.bias_grad_2, &self.layer_2.biases, .{ .alpha = lr * 0.25 });
    }

    pub fn init(self: *@This(), alloc: Allocator) !void {
        try self.layer_1.init(alloc);
        try self.layer_2.init(alloc);

        const shape1 = try brainz.ops.opShape(
            .MatMul,
            self.layer_1.grad.shape,
            try brainz.ops.opShape(.Transpose, self.layer_1.inputShape(), null),
        );

        const shape2 = try brainz.ops.opShape(
            .MatMul,
            self.layer_2.grad.shape,
            try brainz.ops.opShape(.Transpose, self.layer_1.outputShape(), null),
        );

        self.weight_grad_1 = try Tensor(f32).init(shape1, alloc);
        self.weight_grad_2 = try Tensor(f32).init(shape2, alloc);

        self.weight_grad_1_f = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.weight_grad_1.shape, 0), alloc);
        self.weight_grad_2_f = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.weight_grad_2.shape, 0), alloc);

        self.bias_grad_1 = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.outputShape(), 0), alloc);
        self.bias_grad_2 = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.outputShape(), 0), alloc);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const alloc = arena.allocator();
    var mlp: XorMLP = undefined;

    const device = Device.DummyDevice;

    var expected_mat = try Tensor(f32).initFromSlice(mlp.outputShape(), @constCast(@ptrCast(&[_][1]f32{
        [1]f32{0.0},
        [1]f32{1.0},
        [1]f32{1.0},
        [1]f32{0.0},
    })));

    var loss_grad = try Tensor(f32).init(mlp.outputShape(), alloc);

    var input_mat = try Tensor(f32).initFromSlice(mlp.inputShape(), @constCast(@ptrCast(&[_]f32{
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    })));

    try mlp.init(alloc);

    var out = std.io.getStdOut().writer();

    for (0..10_000) |_| {
        const result = try mlp.forward(device, &input_mat);
        const loss = brainz.ops.binaryCrossEntropyLoss(f32, device, result, &expected_mat);

        try brainz.ops.binaryCrossEntropyLossBackprop(f32, device, result, &expected_mat, &loss_grad);
        try device.barrier();

        try mlp.backwards(device, &loss_grad);
        try mlp.step(device, &input_mat, 0.8);
        try out.print("\rloss: {}             ", .{loss});
    }
    const result = try mlp.forward(device, &input_mat);
    try out.print("Outputs: {}", .{result});
}
