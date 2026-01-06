const std = @import("std");
const tensor = @import("tensor.zig");
const prog = @import("plan.zig");

const elemwise_ops = @import("ops/elemwise.zig");
const matmul_ops = @import("ops/matmul.zig");
const loss_ops = @import("ops/loss.zig");
const activations_ops = @import("ops/activations.zig");

const Tensor = tensor.Tensor;
const TensorArena = tensor.TensorArena;
const Dtype = tensor.Dtype;
const Shape = tensor.Shape;
const OpInfo = prog.OpInfo;
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
};

// ======================== Binary element-wise operations ==============================

pub fn add(plan: *ExecutionPlan, a: *const Tensor, b: *const Tensor) !*const Tensor {
    const out_shape = try a.shape.broadcast(b.shape);
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.arena.makeTensor(a.dtype, out_shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try plan.append(&OPS.ADD, &inputs, out, null);
    return out;
}

pub fn sub(plan: *ExecutionPlan, a: *const Tensor, b: *const Tensor) !*const Tensor {
    const out_shape = try a.shape.broadcast(b.shape);
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.arena.makeTensor(a.dtype, out_shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try plan.append(&OPS.SUB, &inputs, out, null);
    return out;
}

pub fn mul(plan: *ExecutionPlan, a: *const Tensor, b: *const Tensor) !*const Tensor {
    const out_shape = try a.shape.broadcast(b.shape);
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.arena.makeTensor(a.dtype, out_shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try plan.append(&OPS.MUL, &inputs, out, null);
    return out;
}

pub fn div(plan: *ExecutionPlan, a: *const Tensor, b: *const Tensor) !*const Tensor {
    const out_shape = try a.shape.broadcast(b.shape);
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.arena.makeTensor(a.dtype, out_shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try plan.append(&OPS.DIV, &inputs, out, null);
    return out;
}

pub fn matmul(plan: *ExecutionPlan, a: *const Tensor, b: *const Tensor) !*const Tensor {
    // only 2D tensors are supported for now
    if (a.shape.n_dimensions != 2 or b.shape.n_dimensions != 2) return error.ShapeMismatch;
    if (a.shape.dimensions[1] != b.shape.dimensions[0]) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const M = a.shape.dimensions[0];
    const K = b.shape.dimensions[1];

    const out = try plan.arena.makeTensor(a.dtype, .fromSlice(&.{ M, K }), a.requires_grad or b.requires_grad);
    try plan.append(&OPS.MATMUL, &.{ a, b }, out, null);
    return out;
}

pub fn batchedMatMul(plan: *ExecutionPlan, a: *const Tensor, b: *const Tensor) !*const Tensor {
    // a: (B, M, N), b: (N, K) -> (B, M, K)
    if (a.shape.n_dimensions != 3 or b.shape.n_dimensions != 2) return error.ShapeMismatch;
    if (a.shape.dimensions[2] != b.shape.dimensions[0]) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const B = a.shape.dimensions[0];
    const M = a.shape.dimensions[1];
    const K = b.shape.dimensions[1];

    const out = try plan.arena.makeTensor(a.dtype, .fromSlice(&.{ B, M, K }), a.requires_grad or b.requires_grad);
    try plan.append(&OPS.BATCHED_MATMUL, &.{ a, b }, out, null);
    return out;
}

pub fn mseLoss(plan: *ExecutionPlan, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try plan.arena.makeTensor(a.dtype, .fromSlice(&.{1}), a.requires_grad or b.requires_grad);
    try plan.append(&OPS.MSE, &.{ a, b }, out, null);
    return out;
}

pub fn relu(plan: *ExecutionPlan, a: *const Tensor) !*const Tensor {
    const out = try plan.arena.makeTensor(a.dtype, a.shape, a.requires_grad);
    try plan.append(&OPS.RELU, &.{a}, out, null);
    return out;
}

// ============================== Tests =====================================

const testing = std.testing;

test "op: add forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, false);
    const b = try plan.createInput("b", .float32, shape, false);

    const c = try add(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(false);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, true);
    const b = try plan.createInput("b", .float32, shape, true);

    const c = try add(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(true);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, false);
    const b = try plan.createInput("b", .float32, shape, false);

    const c = try sub(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(false);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, true);
    const b = try plan.createInput("b", .float32, shape, true);

    const c = try sub(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(true);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, false);
    const b = try plan.createInput("b", .float32, shape, false);

    const c = try mul(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(false);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, true);
    const b = try plan.createInput("b", .float32, shape, true);

    const c = try mul(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(true);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, false);
    const b = try plan.createInput("b", .float32, shape, false);

    const c = try div(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(true);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, true);
    const b = try plan.createInput("b", .float32, shape, true);

    const c = try div(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(true);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shapeA: Shape = comptime .fromSlice(&.{ 2, 3 });
    const a = try plan.createInput("a", .float32, shapeA, false);

    const shapeB: Shape = comptime .fromSlice(&.{ 3, 2 });
    const b = try plan.createInput("b", .float32, shapeB, false);

    // (2,3) * (3,2) gives a (2,2) matrix
    const c = try matmul(&plan, a, b);
    try testing.expectEqual(Shape.fromSlice(&.{ 2, 2 }), c.shape);

    try plan.registerOutput("c", c);

    try plan.finalize(false);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shapeA: Shape = comptime .fromSlice(&.{ 1, 2 });
    const a = try plan.createInput("a", .float32, shapeA, true);

    const shapeB: Shape = comptime .fromSlice(&.{ 2, 1 });
    const b = try plan.createInput("b", .float32, shapeB, true);

    // (1,2) * (2,1) gives (1,1), a dot product basically
    const c = try matmul(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(true);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape: Shape = comptime .fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, false);
    const b = try plan.createInput("b", .float32, shape, false);

    const loss = try mseLoss(&plan, a, b);
    try plan.registerOutput("loss", loss);

    try plan.finalize(false);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape: Shape = comptime .fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shape, true);
    const b = try plan.createInput("b", .float32, shape, true);

    const loss = try mseLoss(&plan, a, b);
    try plan.registerOutput("loss", loss);

    try plan.finalize(true);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    // broadcast [2] -> [2, 2]
    // a: [1.0, 2.0] -> [[1.0, 2.0], [1.0, 2.0]]
    // b: [[3.0, 4.0], [5.0, 6.0]]
    // c: [[4.0, 6.0], [6.0, 8.0]]

    const shapeA = Shape.fromSlice(&.{2});
    const shapeB = Shape.fromSlice(&.{ 2, 2 });
    const a = try plan.createInput("a", .float32, shapeA, false);
    const b = try plan.createInput("b", .float32, shapeB, false);

    const c = try add(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(false);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    // broadcast a: [1] -> b: [2]
    // a: [10.0] -> [10.0, 10.0]
    // b: [1.0, 2.0]
    // c: [11.0, 12.0]
    // grad_c: [1.0, 1.0]
    // grad_a: [1.0 + 1.0] = [2.0]
    // grad_b: [1.0, 1.0]

    const shapeA = Shape.fromSlice(&.{1});
    const shapeB = Shape.fromSlice(&.{2});
    const a = try plan.createInput("a", .float32, shapeA, true);
    const b = try plan.createInput("b", .float32, shapeB, true);

    const c = try add(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(true);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    // a: (2, 2, 2), b: (2, 2) -> out: (2, 2, 2)
    const shapeA = Shape.fromSlice(&.{ 2, 2, 2 });
    const shapeB = Shape.fromSlice(&.{ 2, 2 });
    const a = try plan.createInput("a", .float32, shapeA, false);
    const b = try plan.createInput("b", .float32, shapeB, false);

    const c = try batchedMatMul(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(false);
    try tensorArena.allocateStorage();

    // a[0] = [[1, 2], [3, 4]]
    // a[1] = [[5, 6], [7, 8]]
    @memcpy(a.slice(f32).?, &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 });

    // b = [[1, 0], [0, 1]] (identity)
    @memcpy(b.slice(f32).?, &[_]f32{ 1, 0, 0, 1 });

    try plan.forward();

    // c should be identical to a
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }, c.slice(f32).?);
}

test "op: batched matmul backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    // a: (2, 1, 2), b: (2, 1) -> out: (2, 1, 1)
    const shapeA = Shape.fromSlice(&.{ 2, 1, 2 });
    const shapeB = Shape.fromSlice(&.{ 2, 1 });
    const a = try plan.createInput("a", .float32, shapeA, true);
    const b = try plan.createInput("b", .float32, shapeB, true);

    const c = try batchedMatMul(&plan, a, b);
    try plan.registerOutput("c", c);

    try plan.finalize(true);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{4});
    const a = try plan.createInput("a", .float32, shape, false);

    const b = try relu(&plan, a);
    try plan.registerOutput("b", b);

    try plan.finalize(false);
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

    var plan: ExecutionPlan = .init(&tensorArena, memArena.allocator());
    defer plan.deinit();

    const shape = comptime Shape.fromSlice(&.{4});
    const a = try plan.createInput("a", .float32, shape, true);

    const b = try relu(&plan, a);
    try plan.registerOutput("b", b);

    try plan.finalize(true);
    try tensorArena.allocateStorage();

    @memcpy(a.slice(f32).?, &[_]f32{ -1.0, 0.0, 1.0, 2.0 });

    try plan.forward();

    @memset(b.grad.?.slice(f32).?, 1.0);
    @memset(a.grad.?.slice(f32).?, 0);

    try plan.backward();

    try testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 1.0, 1.0 }, a.grad.?.slice(f32).?);
}
