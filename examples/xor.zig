///! XOR example training a small neural network with ReLU activation to learn the XOR truth table
const std = @import("std");
const brainz = @import("brainz");

const optim = brainz.optim;
const ops = brainz.ops;

const Dtype = brainz.Dtype;
const Shape = brainz.Shape;
const Tensor = brainz.Tensor;
const TensorArena = brainz.TensorArena;
const Linear = brainz.nn.Linear;

const LinearPlan = brainz.LinearPlan;
const ExecutionPlan = brainz.ExecutionPlan;

//TODO: design a comptime API that essentially wraps a net structure that:
// -  implements automatically a forward function based on declaration order
// - a function to save / load weights based on declaration order

/// A small MLP to learn the XOR truth table.
const XorMlp = struct {
    layer_1: Linear(f32, true),
    layer_2: Linear(f32, true),

    pub fn init(plan: *LinearPlan) !@This() {
        return .{
            .layer_1 = try .init(plan, 2, 4),
            .layer_2 = try .init(plan, 4, 1),
        };
    }

    pub fn randomizeWeights(self: *const @This(), rnd: std.Random) void {
        self.layer_1.randomizeWeights(rnd);
        self.layer_2.randomizeWeights(rnd);
    }

    pub fn forward(self: *const @This(), plan: *LinearPlan, input: *const Tensor) !*const Tensor {
        const a = try self.layer_1.forward(plan, input);
        const h1 = try ops.relu(plan, a);
        const b = try self.layer_2.forward(plan, h1);
        return b;
    }
};

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

    // initialize network
    const xorMlp: XorMlp = try .init(&planBuilder);

    // create inputs
    const x = try planBuilder.createInput("x", .float32, x_shape, false);
    const y_target = try planBuilder.createInput("y", .float32, y_shape, false);

    // do forward pass of the network
    const y_pred = try xorMlp.forward(&planBuilder, x);

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

    xorMlp.randomizeWeights(random);

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
