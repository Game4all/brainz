const std = @import("std");
const tensor = @import("tensor.zig");
const prog = @import("plan.zig");

const elemwise_ops = @import("ops/elemwise.zig");
const matmul_ops = @import("ops/matmul.zig");
const loss_ops = @import("ops/loss.zig");
const activations_ops = @import("ops/activations.zig");
const reductions_ops = @import("ops/reductions.zig");

const Tensor = tensor.Tensor;
const TensorArena = tensor.TensorArena;
const Dtype = tensor.Dtype;
const Shape = tensor.Shape;
const OpInfo = prog.OpInfo;

const LinearPlan = prog.LinearPlan;
const PlanBuilder = prog.PlanBuilder;
const ExecutionPlan = prog.ExecutionPlan;

const OPS = struct {
    pub const ADD: OpInfo = .{
        .name = "Add",
        .forward = elemwise_ops.forwardAdd,
        .backward = elemwise_ops.backwardAdd,
    };

    pub const SUB: OpInfo = .{
        .name = "Sub",
        .forward = elemwise_ops.forwardSub,
        .backward = elemwise_ops.backwardSub,
    };

    pub const MUL: OpInfo = .{
        .name = "Mul",
        .forward = elemwise_ops.forwardMul,
        .backward = elemwise_ops.backwardMul,
    };

    pub const DIV: OpInfo = .{
        .name = "Div",
        .forward = elemwise_ops.forwardDiv,
        .backward = elemwise_ops.backwardDiv,
    };

    pub const MATMUL: OpInfo = .{
        .name = "MatMul",
        .forward = matmul_ops.forwardMatMul,
        .backward = matmul_ops.backwardMatMul,
    };

    pub const MSE: OpInfo = .{
        .name = "MSE",
        .forward = loss_ops.forwardMSE,
        .backward = loss_ops.backwardMSE,
    };

    pub const BATCHED_MATMUL: OpInfo = .{
        .name = "BatchedMatMul",
        .forward = matmul_ops.forwardBatchedMatMul,
        .backward = matmul_ops.backwardBatchedMatMul,
    };

    pub const RELU: OpInfo = .{
        .name = "ReLU",
        .forward = activations_ops.forwardReLU,
        .backward = activations_ops.backwardReLU,
    };

    pub const SIGMOID: OpInfo = .{
        .name = "Sigmoid",
        .forward = activations_ops.forwardSigmoid,
        .backward = activations_ops.backwardSigmoid,
    };

    pub const CROSS_ENTROPY: OpInfo = .{
        .name = "CrossEntropy",
        .forward = loss_ops.forwardCrossEntropy,
        .backward = loss_ops.backwardCrossEntropy,
    };

    pub const ARGMAX: OpInfo = .{
        .name = "Argmax",
        .forward = reductions_ops.forwardArgmax,
        .backward = reductions_ops.backwardArgmax,
    };

    pub const SOFTMAX: OpInfo = .{
        .name = "Softmax",
        .forward = activations_ops.forwardSoftmax,
        .backward = activations_ops.backwardSoftmax,
    };
};

// ======================== Binary element-wise operations ==============================

pub fn add(plan: *PlanBuilder, a: *const Tensor, b: *const Tensor) !*const Tensor {
    const out_shape = try a.shape.broadcast(b.shape);
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.createTensor(a.dtype, out_shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try plan.addOp(&OPS.ADD, &inputs, out, null);
    return out;
}

pub fn sub(plan: *PlanBuilder, a: *const Tensor, b: *const Tensor) !*const Tensor {
    const out_shape = try a.shape.broadcast(b.shape);
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.createTensor(a.dtype, out_shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try plan.addOp(&OPS.SUB, &inputs, out, null);
    return out;
}

pub fn mul(plan: *PlanBuilder, a: *const Tensor, b: *const Tensor) !*const Tensor {
    const out_shape = try a.shape.broadcast(b.shape);
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.createTensor(a.dtype, out_shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try plan.addOp(&OPS.MUL, &inputs, out, null);
    return out;
}

pub fn div(plan: *PlanBuilder, a: *const Tensor, b: *const Tensor) !*const Tensor {
    const out_shape = try a.shape.broadcast(b.shape);
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.createTensor(a.dtype, out_shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try plan.addOp(&OPS.DIV, &inputs, out, null);
    return out;
}

pub fn matmul(plan: *PlanBuilder, a: *const Tensor, b: *const Tensor) !*const Tensor {
    // only 2D tensors are supported for now
    if (a.shape.n_dimensions != 2 or b.shape.n_dimensions != 2) return error.ShapeMismatch;
    if (a.shape.dimensions[1] != b.shape.dimensions[0]) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const M = a.shape.dimensions[0];
    const K = b.shape.dimensions[1];

    const out = try plan.createTensor(a.dtype, .fromSlice(&.{ M, K }), a.requires_grad or b.requires_grad);
    try plan.addOp(&OPS.MATMUL, &.{ a, b }, out, null);
    return out;
}

pub fn batchedMatMul(plan: *PlanBuilder, a: *const Tensor, b: *const Tensor) !*const Tensor {
    // a: (B, M, N), b: (N, K) -> (B, M, K)
    if (a.shape.n_dimensions != 3 or b.shape.n_dimensions != 2) return error.ShapeMismatch;
    if (a.shape.dimensions[2] != b.shape.dimensions[0]) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const B = a.shape.dimensions[0];
    const M = a.shape.dimensions[1];
    const K = b.shape.dimensions[1];

    const out = try plan.createTensor(a.dtype, .fromSlice(&.{ B, M, K }), a.requires_grad or b.requires_grad);
    try plan.addOp(&OPS.BATCHED_MATMUL, &.{ a, b }, out, null);
    return out;
}

pub fn mseLoss(plan: *PlanBuilder, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.createTensor(a.dtype, .fromSlice(&.{1}), a.requires_grad or b.requires_grad);
    try plan.addOp(&OPS.MSE, &.{ a, b }, out, null);
    return out;
}

pub fn crossEntropyLoss(plan: *PlanBuilder, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.createTensor(a.dtype, .fromSlice(&.{1}), a.requires_grad or b.requires_grad);
    try plan.addOp(&OPS.CROSS_ENTROPY, &.{ a, b }, out, null);
    return out;
}

pub fn relu(plan: *PlanBuilder, a: *const Tensor) !*const Tensor {
    const out = try plan.createTensor(a.dtype, a.shape, a.requires_grad);
    try plan.addOp(&OPS.RELU, &.{a}, out, null);
    return out;
}

pub fn sigmoid(plan: *PlanBuilder, a: *const Tensor) !*const Tensor {
    const out = try plan.createTensor(a.dtype, a.shape, a.requires_grad);
    try plan.addOp(&OPS.SIGMOID, &.{a}, out, null);
    return out;
}

pub fn softmax(plan: *PlanBuilder, a: *const Tensor, axis: usize) !*const Tensor {
    if (axis >= a.shape.n_dimensions) return error.AxisOutOfBounds;
    const out = try plan.createTensor(a.dtype, a.shape, a.requires_grad);
    try plan.addOp(&OPS.SOFTMAX, &.{a}, out, @ptrFromInt(axis));
    return out;
}

pub fn argMax(plan: *PlanBuilder, a: *const Tensor, axis: usize) !*const Tensor {
    if (axis >= a.shape.n_dimensions) return error.AxisOutOfBounds;

    var outDims: [Shape.MAX_DIMENSIONS]usize = undefined;
    var outNDimensions: usize = 0;
    for (0..a.shape.n_dimensions) |d| {
        if (d == axis) continue;
        outDims[outNDimensions] = a.shape.dimensions[d];
        outNDimensions += 1;
    }

    const outShape = if (outNDimensions == 0)
        Shape.fromSlice(&.{1})
    else
        Shape.fromSlice(outDims[0..outNDimensions]);

    const out = try plan.createTensor(.usize64, outShape, false);
    try plan.addOp(&OPS.ARGMAX, &.{a}, out, @ptrFromInt(axis));
    return out;
}

// ============================== Tests =====================================

const testing = std.testing;

test "op: add forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, false);
    const b = try builder.createInput("b", .float32, shape, false);

    const c = try add(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32) orelse return error.NullSlice;
    @memcpy(aSlice, &[_]f32{ 2.0, 3.0 });
    const bSlice = b.slice(f32) orelse return error.NullSlice;
    @memcpy(bSlice, &[_]f32{ 4.0, 5.0 });

    try plan.forward();

    const cSlice = c.slice(f32) orelse return error.NullSlice;
    try testing.expectEqual(6.0, cSlice[0]);
    try testing.expectEqual(8.0, cSlice[1]);
}

test "op: add backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    errdefer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, true);
    const b = try builder.createInput("b", .float32, shape, true);

    const c = try add(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32) orelse return error.NullSlice;
    const bSlice = b.slice(f32) orelse return error.NullSlice;
    @memcpy(aSlice, &[_]f32{ 2.0, 3.0 });
    @memcpy(bSlice, &[_]f32{ 4.0, 5.0 });

    try plan.forward();

    const cGradSlice = c.grad.?.slice(f32) orelse return error.NullSlice;
    @memcpy(cGradSlice, &[_]f32{ 1.0, 1.0 });
    @memset(a.grad.?.slice(f32) orelse return error.NullSlice, 0);
    @memset(b.grad.?.slice(f32) orelse return error.NullSlice, 0);

    try plan.backward();

    const aGradSlice = a.grad.?.slice(f32) orelse return error.NullSlice;
    const bGradSlice = b.grad.?.slice(f32) orelse return error.NullSlice;
    try testing.expectEqual(1.0, aGradSlice[0]);
    try testing.expectEqual(1.0, bGradSlice[0]);
}

test "op: sub forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    errdefer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, false);
    const b = try builder.createInput("b", .float32, shape, false);

    const c = try sub(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 5.0, 7.0 });

    const bSlice = b.slice(f32).?;
    @memcpy(bSlice, &[_]f32{ 2.0, 3.0 });

    try plan.forward();

    const cSlice = c.slice(f32).?;
    try testing.expectEqualSlices(f32, &[_]f32{ 3.0, 4.0 }, cSlice);
}

test "op: sub backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, true);
    const b = try builder.createInput("b", .float32, shape, true);

    const c = try sub(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 5.0, 7.0 });

    const bSlice = b.slice(f32).?;
    @memcpy(bSlice, &[_]f32{ 2.0, 3.0 });

    try plan.forward();

    const cGradSlice = c.grad.?.slice(f32).?;
    @memset(cGradSlice, 1.0);

    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try plan.backward();

    const aGradSlice = a.grad.?.slice(f32).?;
    const bGradSlice = b.grad.?.slice(f32).?;

    try testing.expectEqual(1.0, aGradSlice[0]);
    try testing.expectEqual(-1.0, bGradSlice[0]);
}

test "op: mul forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, false);
    const b = try builder.createInput("b", .float32, shape, false);

    const c = try mul(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    aSlice[0] = 2.0;
    aSlice[1] = 3.0;
    bSlice[0] = 4.0;
    bSlice[1] = 5.0;

    try plan.forward();

    const cSlice = c.slice(f32).?;
    try testing.expectEqual(8.0, cSlice[0]);
    try testing.expectEqual(15.0, cSlice[1]);
}

test "op: mul backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, true);
    const b = try builder.createInput("b", .float32, shape, true);

    const c = try mul(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    aSlice[0] = 2.0;
    aSlice[1] = 3.0;
    bSlice[0] = 4.0;
    bSlice[1] = 5.0;

    try plan.forward();

    const cGradSlice = c.grad.?.slice(f32).?;
    cGradSlice[0] = 1.0;
    cGradSlice[1] = 1.0;
    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try plan.backward();

    const aGradSlice = a.grad.?.slice(f32).?;
    const bGradSlice = b.grad.?.slice(f32).?;
    try testing.expectEqual(4.0, aGradSlice[0]); // b
    try testing.expectEqual(2.0, bGradSlice[0]); // a
}

test "op: div forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, false);
    const b = try builder.createInput("b", .float32, shape, false);

    const c = try div(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    aSlice[0] = 8.0;
    aSlice[1] = 15.0;
    bSlice[0] = 2.0;
    bSlice[1] = 3.0;

    try plan.forward();

    const cSlice = c.slice(f32).?;
    try testing.expectEqual(4.0, cSlice[0]);
    try testing.expectEqual(5.0, cSlice[1]);
}

test "op: div backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, true);
    const b = try builder.createInput("b", .float32, shape, true);

    const c = try div(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    aSlice[0] = 8.0;
    aSlice[1] = 15.0;
    bSlice[0] = 2.0;
    bSlice[1] = 3.0;

    try plan.forward();

    const cGradSlice = c.grad.?.slice(f32).?;
    cGradSlice[0] = 1.0;
    cGradSlice[1] = 1.0;
    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try plan.backward();

    const aGradSlice = a.grad.?.slice(f32).?;
    const bGradSlice = b.grad.?.slice(f32).?;

    // d(a/b)/da = 1/b = 1/2 = 0.5
    try testing.expectEqual(0.5, aGradSlice[0]);
    try testing.expectEqual(-2.0, bGradSlice[0]);
}

test "op: matmul forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shapeA: Shape = comptime .fromSlice(&.{ 2, 3 });
    const a = try builder.createInput("a", .float32, shapeA, false);

    const shapeB: Shape = comptime .fromSlice(&.{ 3, 2 });
    const b = try builder.createInput("b", .float32, shapeB, false);

    // (2,3) * (3,2) gives a (2,2) matrix
    const c = try matmul(builder, a, b);
    try testing.expectEqual(Shape.fromSlice(&.{ 2, 2 }), c.shape);

    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    // [[1, 2, 3],
    //  [4, 5, 6]]
    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    // [[7, 8],
    //  [9, 1],
    //  [2, 3]]
    const bSlice = b.slice(f32).?;
    @memcpy(bSlice, &[_]f32{ 7.0, 8.0, 9.0, 1.0, 2.0, 3.0 });

    try plan.forward();

    // we should get
    // C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55

    const cSlice = c.slice(f32).?;
    try testing.expectEqualSlices(f32, &[_]f32{ 31.0, 19.0, 85.0, 55.0 }, cSlice);
}

test "op: matmul backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shapeA: Shape = comptime .fromSlice(&.{ 1, 2 });
    const a = try builder.createInput("a", .float32, shapeA, true);

    const shapeB: Shape = comptime .fromSlice(&.{ 2, 1 });
    const b = try builder.createInput("b", .float32, shapeB, true);

    // (1,2) * (2,1) gives (1,1), a dot product basically
    const c = try matmul(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 1.0, 2.0 });

    const bSlice = b.slice(f32).?;
    @memcpy(bSlice, &[_]f32{ 3.0, 4.0 });

    try plan.forward();

    // 3 + 2 * 4 = 11
    const cSlice = c.slice(f32).?;
    try testing.expectEqual(11.0, cSlice[0]);

    // make gradient dC = 1.0 just for simplicity
    const cGradSlice = c.grad.?.slice(f32).?;
    cGradSlice[0] = 1.0;

    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try plan.backward();

    // dA = dC * B^T = 1.0 * [[3,4]] = [[3,4]]
    const aGradSlice = a.grad.?.slice(f32).?;
    try testing.expectEqual(3.0, aGradSlice[0]);
    try testing.expectEqual(4.0, aGradSlice[1]);

    // dB = A^T * dC = [[1],[2]] * 1.0 = [[1],[2]]
    const bGradSlice = b.grad.?.slice(f32).?;
    try testing.expectEqual(1.0, bGradSlice[0]);
    try testing.expectEqual(2.0, bGradSlice[1]);
}

test "op: mse forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape: Shape = comptime .fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, false);
    const b = try builder.createInput("b", .float32, shape, false);

    const loss = try mseLoss(builder, a, b);
    try builder.registerOutput("loss", loss);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 1.0, 2.0 });

    const bSlice = b.slice(f32).?;
    @memcpy(bSlice, &[_]f32{ 3.0, 5.0 });

    // (1-3)^2 = 4
    // (2-5)^2 = 9
    // 9 + 4 = 13
    // mean value = 13 / 2 = 6.5

    try plan.forward();

    const lossScalar = loss.scalar(f32).?;
    try testing.expectEqual(6.5, lossScalar);
}

test "op: mse backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape: Shape = comptime .fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shape, true);
    const b = try builder.createInput("b", .float32, shape, true);

    const loss = try mseLoss(builder, a, b);
    try builder.registerOutput("loss", loss);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 1.0, 2.0 });
    @memcpy(bSlice, &[_]f32{ 3.0, 5.0 });

    try plan.forward();

    const lossGrad = loss.grad.?.slice(f32).?;
    lossGrad[0] = 1.0; // backprop 1.0

    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try plan.backward();

    const aGrad = a.grad.?.slice(f32).?;
    const bGrad = b.grad.?.slice(f32).?;

    // N = 2
    // dL/da = (2/N) * (a - b) * 1.0 = (a - b)
    // a[0] - b[0] = 1 - 3 = -2
    // a[1] - b[1] = 2 - 5 = -3

    try testing.expectEqual(-2.0, aGrad[0]);
    try testing.expectEqual(-3.0, aGrad[1]);

    // dL/db = -(2/N) * (a - b) = - (dL/da)
    try testing.expectEqual(2.0, bGrad[0]);
    try testing.expectEqual(3.0, bGrad[1]);
}

test "op: add broadcasting forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    // broadcast [2] -> [2, 2]
    // a: [1.0, 2.0] -> [[1.0, 2.0], [1.0, 2.0]]
    // b: [[3.0, 4.0], [5.0, 6.0]]
    // c: [[4.0, 6.0], [6.0, 8.0]]

    const shapeA = Shape.fromSlice(&.{2});
    const shapeB = Shape.fromSlice(&.{ 2, 2 });
    const a = try builder.createInput("a", .float32, shapeA, false);
    const b = try builder.createInput("b", .float32, shapeB, false);

    const c = try add(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 1.0, 2.0 });

    const bSlice = b.slice(f32).?;
    @memcpy(bSlice, &[_]f32{ 3.0, 4.0, 5.0, 6.0 });

    try plan.forward();

    const cSlice = c.slice(f32).?;
    try testing.expectEqualSlices(f32, &[_]f32{ 4.0, 6.0, 6.0, 8.0 }, cSlice);
}

test "op: add broadcasting backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    // broadcast a: [1] -> b: [2]
    // a: [10.0] -> [10.0, 10.0]
    // b: [1.0, 2.0]
    // c: [11.0, 12.0]
    // grad_c: [1.0, 1.0]
    // grad_a: [1.0 + 1.0] = [2.0]
    // grad_b: [1.0, 1.0]

    const shapeA = Shape.fromSlice(&.{1});
    const shapeB = Shape.fromSlice(&.{2});
    const a = try builder.createInput("a", .float32, shapeA, true);
    const b = try builder.createInput("b", .float32, shapeB, true);

    const c = try add(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    a.slice(f32).?[0] = 10.0;
    @memcpy(b.slice(f32).?, &[_]f32{ 1.0, 2.0 });

    try plan.forward();

    const cGradSlice = c.grad.?.slice(f32).?;
    @memset(cGradSlice, 1.0);

    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try plan.backward();

    try testing.expectEqual(2.0, a.grad.?.slice(f32).?[0]);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 1.0 }, b.grad.?.slice(f32).?);
}

test "op: batched matmul forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    // a: (2, 2, 2), b: (2, 2) -> out: (2, 2, 2)
    const shapeA = Shape.fromSlice(&.{ 2, 2, 2 });
    const shapeB = Shape.fromSlice(&.{ 2, 2 });
    const a = try builder.createInput("a", .float32, shapeA, false);
    const b = try builder.createInput("b", .float32, shapeB, false);

    const c = try batchedMatMul(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    // a[0] = [[1, 2], [3, 4]]
    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 });

    // b = [[1, 2], [3, 4]]
    const bSlice = b.slice(f32).?;
    @memcpy(bSlice, &[_]f32{ 1, 2, 3, 4 });

    try plan.forward();

    const cSlice = c.slice(f32).?;
    // result [0] = [[1, 2], [3, 4]] * [[1, 2], [3, 4]] = [[7, 10], [15, 22]]
    // result [1] = [[5, 6], [7, 8]] * [[1, 2], [3, 4]] = [[23, 34], [31, 46]]
    try testing.expectEqualSlices(f32, &[_]f32{ 7, 10, 15, 22, 23, 34, 31, 46 }, cSlice);
}

test "op: cross_entropy forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape: Shape = comptime .fromSlice(&.{3});

    const pred = try builder.createInput("pred", .float32, shape, false); // predicted class probs
    const target = try builder.createInput("target", .float32, shape, false); // target class probs

    const loss = try crossEntropyLoss(&linearPlan.builder, pred, target);
    try builder.registerOutput("loss", loss);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const predSlice = pred.slice(f32).?;
    @memcpy(predSlice, &[_]f32{ 0.1, 0.7, 0.2 });

    const targetSlice = target.slice(f32).?;
    @memcpy(targetSlice, &[_]f32{ 0.0, 1.0, 0.0 });

    // loss = -(0.0 * log(0.1) + 1.0 * log(0.7) + 0.0 * log(0.2)) / 3
    // loss = -log(0.7) / 3 = 0.35667 / 3 = 0.11889

    try plan.forward();

    const lossScalar = loss.scalar(f32).?;
    try testing.expectApproxEqAbs(@as(f32, 0.11889), lossScalar, 1e-5);
}

test "op: batched matmul backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    // a: (2, 1, 2), b: (2, 1) -> out: (2, 1, 1)
    const shapeA = Shape.fromSlice(&.{ 2, 1, 2 });
    const shapeB = Shape.fromSlice(&.{ 2, 1 });
    const a = try builder.createInput("a", .float32, shapeA, true);
    const b = try builder.createInput("b", .float32, shapeB, true);

    const c = try batchedMatMul(builder, a, b);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    // a[0] = [[1, 2]], a[1] = [[3, 4]]
    @memcpy(a.slice(f32).?, &[_]f32{ 1, 2, 3, 4 });
    // b = [[5], [6]]
    @memcpy(b.slice(f32).?, &[_]f32{ 5, 6 });

    try plan.forward();

    // c[0] = 1*5 + 2*6 = 17
    // c[1] = 3*5 + 4*6 = 15 + 24 = 39
    try testing.expectEqual(17.0, c.slice(f32).?[0]);
    try testing.expectEqual(39.0, c.slice(f32).?[1]);

    // dC = [[1], [1]]
    @memset(c.grad.?.slice(f32).?, 1.0);
    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try plan.backward();

    // dA[b, m, n] = sum_k (dOut[b, m, k] * B[n, k])
    // dA[0, 0, 0] = dC[0,0,0] * B[0, 0] = 1 * 5 = 5
    // dA[0, 0, 1] = dC[0,0,0] * B[1, 0] = 1 * 6 = 6
    // dA[1, 0, 0] = dC[1,0,0] * B[0, 0] = 1 * 5 = 5
    // dA[1, 0, 1] = dC[1,0,0] * B[1, 0] = 1 * 6 = 6
    try testing.expectEqualSlices(f32, &[_]f32{ 5, 6, 5, 6 }, a.grad.?.slice(f32).?);

    // dB[n, k] = sum_b,m (dOut[b, m, k] * A[b, m, n])
    // dB[0, 0] = dC[0,0,0]*A[0,0,0] + dC[1,0,0]*A[1,0,0] = 1*1 + 1*3 = 4
    // dB[1, 0] = dC[0,0,0]*A[0,0,1] + dC[1,0,0]*A[1,0,1] = 1*2 + 1*4 = 6
    try testing.expectEqualSlices(f32, &[_]f32{ 4, 6 }, b.grad.?.slice(f32).?);
}

test "op: relu forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{4});
    const a = try builder.createInput("a", .float32, shape, false);

    const b = try relu(builder, a);
    try builder.registerOutput("b", b);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ -1.0, 0.0, 1.0, 2.0 });

    try plan.forward();

    const bSlice = b.slice(f32).?;
    try testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 1.0, 2.0 }, bSlice);
}

test "op: relu backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{4});
    const a = try builder.createInput("a", .float32, shape, true);

    const b = try relu(builder, a);
    try builder.registerOutput("b", b);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    @memcpy(a.slice(f32).?, &[_]f32{ -1.0, 0.0, 1.0, 2.0 });

    try plan.forward();

    @memset(b.grad.?.slice(f32).?, 1.0);
    @memset(a.grad.?.slice(f32).?, 0);

    try plan.backward();

    try testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 1.0, 1.0 }, a.grad.?.slice(f32).?);
}

test "op: sigmoid forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{3});
    const a = try builder.createInput("a", .float32, shape, false);

    const b = try sigmoid(builder, a);
    try builder.registerOutput("b", b);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;

    // sigmoid(0) = 0.5
    // sigmoid(+inf) -> 1.0
    // sigmoid(-inf) -> 0.0
    @memcpy(aSlice, &[_]f32{ 0.0, 100.0, -100.0 });

    try plan.forward();

    const bSlice = b.slice(f32).?;
    try testing.expectApproxEqAbs(@as(f32, 0.5), bSlice[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), bSlice[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), bSlice[2], 1e-5);
}

test "op: sigmoid backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{1});
    const a = try builder.createInput("a", .float32, shape, true);

    const b = try sigmoid(builder, a);
    try builder.registerOutput("b", b);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    // sigmoid(0) = 0.5
    a.slice(f32).?[0] = 0.0;

    try plan.forward();

    // dSigmoid(0) / dx = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    b.grad.?.slice(f32).?[0] = 1.0;
    @memset(a.grad.?.slice(f32).?, 0);

    try plan.backward();

    try testing.expectApproxEqAbs(@as(f32, 0.25), a.grad.?.slice(f32).?[0], 1e-5);
}

test "op: argmax forward axis (2 dimensions)" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = comptime Shape.fromSlice(&.{ 2, 3 });
    const a = try builder.createInput("a", .float32, shape, false);

    // argMax on the innermost dimension -> shape (2)
    const c = try argMax(&linearPlan.builder, a, 1);
    try builder.registerOutput("c", c);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    // [[1, 5, 2],
    //  [4, 3, 6]]
    @memcpy(aSlice, &[_]f32{ 1.0, 5.0, 2.0, 4.0, 3.0, 6.0 });

    try plan.forward();

    const cSlice = c.slice(u64).?;
    try testing.expectEqual(@as(u64, 1), cSlice[0]); // argmax([1, 5, 2]) = 5 at index 1
    try testing.expectEqual(@as(u64, 2), cSlice[1]); // argmax([4, 3, 6]) = 6 at index 2
}

test "op: softmax forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = Shape.fromSlice(&.{ 2, 3 });
    const a = try builder.createInput("a", .float32, shape, false);

    const s = try softmax(&linearPlan.builder, a, 1);
    try builder.registerOutput("s", s);

    var plan = try linearPlan.finalize(false);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 1.0, 2.0, 3.0, 0.0, 0.0, 0.0 });

    try plan.forward();

    const sSlice = s.slice(f32).?;
    // row 0: [1, 2, 3]
    // sum = e^1 + e^2 + e^3 = 30.193
    // s[0] = e^1 / sum ~= 0.09003
    // s[1] = e^2 / sum ~= 0.24473
    // s[2] = e^3 / sum ~= 0.66524
    try testing.expectApproxEqAbs(@as(f32, 0.09003), sSlice[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.24473), sSlice[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.66524), sSlice[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sSlice[0] + sSlice[1] + sSlice[2], 1e-5);

    // row 1: [0, 0, 0] -> [1/3, 1/3, 1/3]
    try testing.expectApproxEqAbs(@as(f32, 0.33333), sSlice[3], 1e-5);
}

test "op: softmax backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = Shape.fromSlice(&.{ 1, 2 });
    const a = try builder.createInput("a", .float32, shape, true);

    const s = try softmax(&linearPlan.builder, a, 1);
    try builder.registerOutput("s", s);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 0.0, 0.0 });

    try plan.forward();

    const sGradSlice = s.grad.?.slice(f32).?;
    @memcpy(sGradSlice, &[_]f32{ 1.0, 0.0 });
    @memset(a.grad.?.slice(f32).?, 0);

    try plan.backward();

    const aGradSlice = a.grad.?.slice(f32).?;
    // dL/dx0 = s0 * (dL/ds0 - (dL/ds0*s0 + dL/ds1*s1))
    // s0 = 0.5, s1 = 0.5
    // dL/ds0 = 1, dL/ds1 = 0
    // dot = 1 * 0.5 + 0 * 0.5 = 0.5
    // dL/dx0 = 0.5 * (1 - 0.5) = 0.25
    // dL/dx1 = 0.5 * (0 - 0.5) = -0.25

    try testing.expectApproxEqAbs(@as(f32, 0.25), aGradSlice[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -0.25), aGradSlice[1], 1e-5);
}

test "op: softmax + cross_entropy compatibility" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var linearPlan: LinearPlan = .init(&tensorArena, memArena.allocator());
    defer linearPlan.deinit();
    const builder = &linearPlan.builder;

    const shape = Shape.fromSlice(&.{ 1, 3 });
    const logits = try builder.createInput("logits", .float32, shape, true);
    const targets = try builder.createInput("targets", .float32, shape, false);

    const probs = try softmax(builder, logits, 1);
    const loss = try crossEntropyLoss(builder, probs, targets);
    try builder.registerOutput("loss", loss);

    var plan = try linearPlan.finalize(true);
    defer plan.deinit();

    try tensorArena.allocateStorage();

    const logitsSlice = logits.slice(f32).?;
    @memcpy(logitsSlice, &[_]f32{ 1.0, 2.0, 3.0 });
    const targetsSlice = targets.slice(f32).?;
    @memcpy(targetsSlice, &[_]f32{ 0.0, 1.0, 0.0 }); // target class is index 1

    try plan.forward();

    const lossGrad = loss.grad.?.slice(f32).?;
    lossGrad[0] = 1.0;
    @memset(logits.grad.?.slice(f32).?, 0);

    try plan.backward();

    const probsSlice = probs.slice(f32).?;
    const logitsGradSlice = logits.grad.?.slice(f32).?;
    const N: f32 = @floatFromInt(probsSlice.len);

    for (0..3) |i| {
        const expected = (probsSlice[i] - targetsSlice[i]) / N;
        try testing.expectApproxEqAbs(expected, logitsGradSlice[i], 1e-5);
    }
}
