///! Linear regression example where a weight tensor is optimized with SGD to predict points following y = 2x + 1 equation curve
const std = @import("std");
const brainz = @import("brainz");

const optim = brainz.optim;
const ops = brainz.ops;

const Dtype = brainz.Dtype;
const Shape = brainz.Shape;
const Tensor = brainz.Tensor;
const TensorArena = brainz.TensorArena;
const Program = brainz.Program;

pub fn main() !void {
    var gpa = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gpa.deinit();

    const allocator = gpa.allocator();

    var tensorArena: TensorArena = .init(allocator);
    defer tensorArena.deinit();

    const N: usize = 100; // number of samples
    const lr: f32 = 0.01;
    const epochs: usize = 2000;

    // we use augmented matrices to represent the input x with the bias term
    const x_shape: Shape = comptime .fromSlice(&.{ N, 2 });
    const y_shape: Shape = comptime .fromSlice(&.{ N, 1 });
    const w_shape: Shape = comptime .fromSlice(&.{ 2, 1 }); // weight tensor shape

    // define program and linear regression model
    var prog: Program = .init(&tensorArena, allocator);
    defer prog.deinit();

    // create X and target Y inputs (which both are not trainable hence do not require gradients)
    const x = try prog.createInput("x", .float32, x_shape, false);
    const y_target = try prog.createInput("y", .float32, y_shape, false);

    // create a tensor for model parameters (which are trainable)
    const w = try prog.createParam(.float32, w_shape);

    // y_pred = x @ w which is a dot product
    const y_pred = try ops.matmul(&prog, x, w);

    // loss = mse(y_pred, y_target)
    const loss = try ops.mseLoss(&prog, y_pred, y_target);
    try prog.registerOutput("loss", loss);

    // finalize program and allocate backing memory for tensors
    try prog.finalize(true);
    try tensorArena.allocateStorage();

    // initialize default PRNG
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // populate the training data
    const x_data = x.slice(f32).?;
    const y_data = y_target.slice(f32).?;

    for (0..N) |i| {
        const val = random.float(f32) * 10.0;
        x_data[i * 2] = val; // x feature
        x_data[i * 2 + 1] = 1.0; // bias term

        // y = 2x + 1 + noise
        const noise = (random.float(f32) - 0.5) * 0.5;
        y_data[i] = 2.0 * val + 1.0 + noise;
    }

    // initialize weights randomly
    const w_data = w.slice(f32).?;
    for (w_data) |*w_i|
        w_i.* = random.floatNorm(f32);

    // compute initial MSE for display
    try prog.forward();
    const initialLoss = loss.scalar(f32).?;

    std.log.info("Initial weights: w={d:.4}, b={d:.4}, MSE={d:.4}", .{ w_data[0], w_data[1], initialLoss });

    var sgd = optim.SGD.init(prog.getParams(), lr);
    const loss_grad = loss.grad.?.slice(f32).?;

    for (0..epochs) |epoch| {
        try prog.forward();

        const current_loss = loss.scalar(f32).?;
        std.log.info("Epoch {}: Loss = {d:.6}", .{ epoch, current_loss });

        // zero out the gradients
        prog.zeroGrad();
        loss_grad[0] = 1.0; // seed the initial gradient
        try prog.backward();

        sgd.step();
    }

    // compute final MSE for display
    try prog.forward();
    const finalLoss = loss.scalar(f32).?;

    // Result
    std.log.info("Final weights: w={d:.4}, b={d:.4}, Final MSE={d:.4}", .{ w_data[0], w_data[1], finalLoss });
}
