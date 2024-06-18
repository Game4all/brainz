const std = @import("std");
const brainz = @import("brainz");
const Matrix = brainz.Matrix;
const Allocator = std.mem.Allocator;

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

const XorMLP = struct {
    layer_1: brainz.Dense(2, 2, brainz.activation.Sigmoid) = undefined,
    layer_2: brainz.Dense(2, 1, brainz.activation.Sigmoid) = undefined,

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

        brainz.ops.mulScalar(f32, &self.layer_1.grad, lr, &self.layer_1.grad);
        brainz.ops.mulScalar(f32, &self.layer_2.grad, lr, &self.layer_2.grad);

        // compute the actual gradients wrt to the weights of the layers for weight update.
        brainz.ops.matMul(f32, &self.layer_1.grad, &layer1_inputs, &self.weight_grad_1);
        brainz.ops.matMul(f32, &self.layer_2.grad, &layer2_inputs, &self.weight_grad_2);

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

        const shape1 = try brainz.ops.opShape(
            .MatMul,
            self.layer_1.grad.shape,
            .{ 1, 2 },
        );

        const shape2 = try brainz.ops.opShape(
            .MatMul,
            self.layer_2.grad.shape,
            .{ 1, 2 },
        );

        self.weight_grad_1 = try Matrix(f32).empty(shape1, alloc);
        self.weight_grad_2 = try Matrix(f32).empty(shape2, alloc);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const alloc = arena.allocator();
    var mlp: XorMLP = undefined;

    const BCE = brainz.loss.BinaryCrossEntropy;

    var expected_mat = try Matrix(f32).empty(.{ 1, 1 }, alloc);
    var loss_grad = try Matrix(f32).empty(.{ 1, 1 }, alloc);

    var input_mat = try Matrix(f32).empty(.{ 2, 1 }, alloc);

    try mlp.init(alloc);

    var out = std.io.getStdOut().writer();

    for (0..50_000) |_| {
        for (inputs, outputs) |i, o| {
            input_mat.setData(@constCast(&i));
            expected_mat.setData(@constCast(&o));

            const result = mlp.forward(&input_mat);

            // const loss = BCE.compute(result, &expected_mat);
            BCE.computeDerivative(result, &expected_mat, &loss_grad);
            mlp.backwards(&loss_grad);
            mlp.step(&input_mat, 0.1);
        }
    }

    for (inputs, outputs) |i, o| {
        input_mat.setData(@constCast(&i));
        expected_mat.setData(@constCast(&o));

        _ = mlp.forward(&input_mat);
        // const loss = BCE.compute(result, &expected_mat);
    }

    try out.print("Weights: {}", .{mlp.layer_1.weights});
}
