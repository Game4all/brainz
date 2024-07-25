const std = @import("std");
const brainz = @import("brainz");

const Mat = brainz.Tensor;
const Device = brainz.Device;

pub fn main() !void {
    var heap = std.heap.HeapAllocator.init();
    defer heap.deinit();

    var arena = std.heap.ArenaAllocator.init(heap.allocator());
    defer arena.deinit();

    const alloc = arena.allocator();

    var dense: brainz.Dense(1, 1, 4, brainz.activation.Linear) = undefined;
    try dense.init(alloc);

    const loss = brainz.loss.MSE;

    var out = std.io.getStdOut().writer();

    // contains the expected values for backprop
    // outputs follow f(x)=2x + 1
    var expected_mat = try Mat(f32).initFromSlice(dense.outputShape(), @constCast(@ptrCast(&[_][1]f32{
        [1]f32{1.0},
        [1]f32{3.0},
        [1]f32{7.0},
        [1]f32{19.0},
    })));
    // contains the computed loss gradient
    var loss_grad = try Mat(f32).init(dense.outputShape(), alloc);

    // contains the input of the network
    var inputs = try Mat(f32).initFromSlice(dense.inputShape(), @constCast(@ptrCast(&[_][1]f32{
        [1]f32{0.0},
        [1]f32{1.0},
        [1]f32{3.0},
        [1]f32{9.0},
    })));
    var inputsT = inputs.transpose();

    // contains the gradient wrt to the weights
    var weights_grad = try Mat(f32).init(try brainz.ops.opShape(.MatMul, dense.grad.shape, inputsT.shape), alloc);

    // holds the summed batched error gradients for the biases
    var bias_grad_summed = try Mat(f32).init(try brainz.ops.opShape(.Reduce, dense.grad.shape, 0), alloc);
    var weights_grad_summed = try Mat(f32).init(try brainz.ops.opShape(.Reduce, weights_grad.shape, 0), alloc);

    // train for 100 epochs.
    for (0..200) |_| {
        const result = dense.forward(&inputs);
        const loss_val = loss.compute(result, &expected_mat);
        loss.computeDerivative(result, &expected_mat, &loss_grad);

        // compute the gradients for the layer.
        // they are stored in the `.grad` field.
        _ = dense.backwards(&loss_grad);

        // compute the batched gradients wrt to the weights.
        brainz.ops.matMul(f32, Device.DummyDevice, &dense.grad, &inputsT, &weights_grad);

        // sum the batched gradients
        brainz.ops.reduce(f32, .Sum, &dense.grad, 0, &bias_grad_summed);
        brainz.ops.reduce(f32, .Sum, &weights_grad, 0, &weights_grad_summed);

        // update the weights
        brainz.ops.sub(f32, &dense.weights, &weights_grad_summed, &dense.weights, .{ .alpha = 0.05 * 0.25 }); // Wnew = Wold - Wgrad;
        // update the bias
        brainz.ops.sub(f32, &dense.biases, &bias_grad_summed, &dense.biases, .{ .alpha = 0.05 * 0.25 }); // Bnew = Bold - grad;

        try out.print("\rloss: {}                   ", .{loss_val});
    }

    try out.print("\n==== Model Evaluation ===\n", .{});

    const results = dense.forward(&inputs);
    try out.print("outputs: {}", .{results});
}
