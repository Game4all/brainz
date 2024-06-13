const std = @import("std");
const brainz = @import("brainz");

const Mat = brainz.Matrix;

const inputs = [_][1]f32{
    [1]f32{0.0},
    [1]f32{1.0},
    [1]f32{3.0},
    [1]f32{9.0},
};

// outputs follow f(x)=2x + 1
const outputs = [_][1]f32{
    [1]f32{1.0},
    [1]f32{3.0},
    [1]f32{7.0},
    [1]f32{19.0},
};

pub fn main() !void {
    var heap = std.heap.HeapAllocator.init();
    defer heap.deinit();

    var arena = std.heap.ArenaAllocator.init(heap.allocator());
    defer arena.deinit();

    const alloc = arena.allocator();

    var dense: brainz.Dense(1, 1, brainz.activation.Linear) = undefined;
    try dense.init(alloc);

    const loss = brainz.loss.MSE;

    var out = std.io.getStdOut().writer();

    // contains the expected value for backprop
    var expected_mat = try Mat(f32).empty(.{ 1, 1 }, alloc);
    // contains the computed loss gradient
    var loss_grad = try Mat(f32).empty(.{ 1, 1 }, alloc);

    // contains the input of the network
    var input_mat = try Mat(f32).empty(.{ 1, 1 }, alloc);
    var input_transposed = input_mat.transpose();

    // contains the gradient wrt to the weights
    var weights_grad = try Mat(f32).empty(.{ 1, 1 }, alloc);

    // train for 100 epochs.
    for (0..100) |_| {
        for (inputs, outputs) |i, o| {
            @memcpy(input_mat.get_mut_slice(), @constCast(&i));
            @memcpy(expected_mat.get_mut_slice(), @constCast(&o));

            const result = dense.forward(&input_mat);
            const loss_val = loss.compute(result, &expected_mat);
            loss.compute_derivative(result, &expected_mat, &loss_grad);

            // compute the gradients for the layer.
            // they are stored in the `.grad` field.
            _ = dense.backwards(&loss_grad);
            brainz.ops.mul_scalar(f32, &dense.grad, 0.1, &dense.grad); // scale the error gradient by 0.1 so we don't have to do it twice for the weight and bias update.

            // compute the grad wrt to the weights.
            brainz.ops.mul(f32, &dense.grad, &input_transposed, &weights_grad);

            // update the weights
            brainz.ops.sub(f32, &dense.weights, &weights_grad, &dense.weights); // Wnew = Wold - Wgrad;
            // update the bias
            brainz.ops.sub(f32, &dense.biases, &dense.grad, &dense.biases); // Bnew = Bold - grad;

            try out.print("\rloss: {}\t\t", .{loss_val});
        }
    }

    try out.print("\rTraining done.                   \n", .{});

    for (inputs, outputs) |i, o| {
        @memcpy(input_mat.get_mut_slice(), @constCast(&i));
        @memcpy(expected_mat.get_mut_slice(), @constCast(&o));

        const result = dense.forward(&input_mat);
        try out.print("output: {} | expected: {} \n", .{ result.get(.{ 0, 0 }), expected_mat.get(.{ 0, 0 }) });
    }
}
