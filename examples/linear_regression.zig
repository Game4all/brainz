///! Linear regression example where a weight and a bias are optimized with SGD to predict points following y = 2x + 1 equation curve
const std = @import("std");
const brainz = @import("brainz");

const optim = brainz.optim;
const ops = brainz.ops;

const Dtype = brainz.Dtype;
const Shape = brainz.Shape;
const Tensor = brainz.Tensor;
const TensorArena = brainz.TensorArena;
const ExecutionPlan = brainz.ExecutionPlan;

pub fn main() !void {
    var gpa = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gpa.deinit();

    const allocator = gpa.allocator();

    var tensorArena: TensorArena = .init(allocator);
    defer tensorArena.deinit();

    const N: usize = 2500; // number of samples
    const lr: f32 = 0.01;
    const epochs: usize = 2000;

    const x_shape: Shape = comptime .fromSlice(&.{ N, 1 });
    const y_shape: Shape = comptime .fromSlice(&.{ N, 1 });

    // we use separate tensors for weight and bias, which are basically scalars in that case
    const w_shape: Shape = comptime .fromSlice(&.{1});
    const b_shape: Shape = comptime .fromSlice(&.{1});

    // define plan and linear regression model
    var plan: ExecutionPlan = .init(&tensorArena, allocator);
    defer plan.deinit();

    // create X and target Y inputs
    const x = try plan.createInput("x", .float32, x_shape, false);
    const y_target = try plan.createInput("y", .float32, y_shape, false);

    // create tensors for model parameters (trainable)
    const w = try plan.createParam(.float32, w_shape);
    const b = try plan.createParam(.float32, b_shape);

    // y_pred = x * w + b
    const xw = try ops.mul(&plan, x, w);
    const y_pred = try ops.add(&plan, xw, b);

    // loss = mse(y_pred, y_target)
    const loss = try ops.mseLoss(&plan, y_pred, y_target);
    try plan.registerOutput("loss", loss);

    // finalize plan and allocate backing memory for tensors
    try plan.finalize(true);
    try tensorArena.allocateStorage();

    // initialize default PRNG
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // populate the training data
    const x_data = x.slice(f32).?;
    const y_data = y_target.slice(f32).?;

    for (0..N) |i| {
        const val = random.float(f32) * 10.0;
        x_data[i] = val; // x feature

        // y = 2x + 1 + noise
        const noise = (random.float(f32) - 0.5) * 0.5;
        y_data[i] = 2.0 * val + 1.0 + noise;
    }

    // initialize weights and bias randomly
    const w_data = w.slice(f32).?;
    for (w_data) |*w_i|
        w_i.* = random.floatNorm(f32);

    const b_data = b.slice(f32).?;
    for (b_data) |*b_i|
        b_i.* = random.floatNorm(f32);

    // compute initial MSE for display
    try plan.forward();
    const initialLoss = loss.scalar(f32).?;

    std.log.info("Initial weights: w={d:.4}, b={d:.4}, MSE={d:.4}", .{ w_data[0], b_data[0], initialLoss });

    var sgd = optim.SGD.init(plan.getParams(), lr);
    const loss_grad = loss.grad.?.slice(f32).?;

    for (0..epochs) |epoch| {
        try plan.forward();

        const current_loss = loss.scalar(f32).?;
        if (epoch % 100 == 0)
            std.log.info("Epoch {}: Loss = {d:.6}", .{ epoch, current_loss });

        // zero out the gradients
        plan.zeroGrad();
        loss_grad[0] = 1.0; // seed the initial gradient
        try plan.backward();

        sgd.step();
    }

    // compute final MSE for display
    try plan.forward();
    const finalLoss = loss.scalar(f32).?;

    // Result
    std.log.info("Final weights: w={d:.4}, b={d:.4}, Final MSE={d:.4}", .{ w_data[0], b_data[0], finalLoss });
}
