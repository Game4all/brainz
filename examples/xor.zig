///! XOR example training a small neural network with ReLU activation to learn the XOR truth table
const std = @import("std");
const brainz = @import("brainz");

const optim = brainz.optim;
const ops = brainz.ops;

const Dtype = brainz.Dtype;
const Shape = brainz.Shape;
const Tensor = brainz.Tensor;
const TensorArena = brainz.TensorArena;

const LinearPlan = brainz.LinearPlan;
const ExecutionPlan = brainz.ExecutionPlan;

pub fn main() !void {
    var gpa = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gpa.deinit();

    const allocator = gpa.allocator();

    // create the tensor arena
    var tensorArena: TensorArena = .init(allocator);
    defer tensorArena.deinit();

    const lr: f32 = 0.1;
    const epochs: usize = 1000;

    // XOR data: 4 samples, 1x2 feature vector each
    const x_shape: Shape = comptime .fromSlice(&.{ 4, 2 });
    const y_shape: Shape = comptime .fromSlice(&.{ 4, 1 });

    // create a linear execution plan
    var planBuilder: LinearPlan = .init(&tensorArena, allocator);
    errdefer planBuilder.deinit();

    // layer 1
    // 2 inputs -> 4 hidden neurons
    const w1_shape: Shape = comptime .fromSlice(&.{ 2, 4 });
    const b1_shape: Shape = comptime .fromSlice(&.{4});

    // layer 2
    // 4 hidden -> 1 output
    const w2_shape: Shape = comptime .fromSlice(&.{ 4, 1 });
    const b2_shape: Shape = comptime .fromSlice(&.{1});

    // create inputs
    const x = try planBuilder.createInput("x", .float32, x_shape, false);
    const y_target = try planBuilder.createInput("y", .float32, y_shape, false);

    // create weights and biases
    const w1 = try planBuilder.createParam(.float32, w1_shape);
    const b1 = try planBuilder.createParam(.float32, b1_shape);
    const w2 = try planBuilder.createParam(.float32, w2_shape);
    const b2 = try planBuilder.createParam(.float32, b2_shape);

    // implement the model
    // h1 = relu(x @ w1 + b1)
    const xw1 = try ops.matmul(&planBuilder, x, w1);
    const z1 = try ops.add(&planBuilder, xw1, b1);
    const h1 = try ops.relu(&planBuilder, z1);

    // y_pred = h1 @ w2 + b2
    const hw2 = try ops.matmul(&planBuilder, h1, w2);
    const y_pred = try ops.add(&planBuilder, hw2, b2);

    // loss = mse(y_pred, y_target)
    const loss = try ops.mseLoss(&planBuilder, y_pred, y_target);
    try planBuilder.registerOutput("loss", loss);

    // finalize the plan and allocate storage
    var plan = try planBuilder.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    // Initialize data
    const x_data = x.slice(f32).?;
    const y_data = y_target.slice(f32).?;

    @memcpy(x_data, &[_]f32{ 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0 });
    @memcpy(y_data, &[_]f32{ 0.0, 1.0, 1.0, 0.0 });

    // initialize the weights randomly
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (w1.slice(f32).?) |*val| val.* = random.floatNorm(f32) * 0.1;
    for (b1.slice(f32).?) |*val| val.* = random.floatNorm(f32) * 0.1;
    for (w2.slice(f32).?) |*val| val.* = random.floatNorm(f32) * 0.1;
    for (b2.slice(f32).?) |*val| val.* = random.floatNorm(f32) * 0.1;

    // initialize the optimizer
    var sgd = optim.SGD.init(plan.getParams(), lr);
    const loss_grad = loss.grad.?.slice(f32).?;

    std.log.info("=================================== Training =================================== ", .{});

    for (0..epochs) |epoch| {
        try plan.forward();

        const current_loss = loss.scalar(f32).?;
        if (epoch % 100 == 0) {
            std.log.info("Epoch {}: Loss = {d:.6}", .{ epoch, current_loss });
        }

        plan.zeroGrad();
        // seed the loss gradient
        loss_grad[0] = 1.0;
        try plan.backward();

        sgd.step();
    }

    std.log.info("=================================== Evaluation ===================================", .{});

    // Evaluation
    try plan.forward();
    const final_results = y_pred.slice(f32).?;
    std.log.info("0 XOR 0 = {d:.4} (real: 0.0)", .{final_results[0]});
    std.log.info("0 XOR 1 = {d:.4} (real: 1.0)", .{final_results[1]});
    std.log.info("1 XOR 0 = {d:.4} (real: 1.0)", .{final_results[2]});
    std.log.info("1 XOR 1 = {d:.4} (real: 0.0)", .{final_results[3]});
}
