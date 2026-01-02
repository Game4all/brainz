const std = @import("std");
const tensor = @import("tensor.zig");
const prog = @import("program.zig");
const elemwise_ops = @import("ops/elemwise.zig");
const matmul_ops = @import("ops/matmul.zig");

const Tensor = tensor.Tensor;
const TensorArena = tensor.TensorArena;
const Dtype = tensor.Dtype;
const Shape = tensor.Shape;
const OpInfo = prog.OpInfo;
const Program = prog.Program;

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
};

// ======================== Binary element-wise operations ==============================

pub fn add(program: *Program, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try program.arena.makeTensor(a.dtype, a.shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try program.append(&OPS.ADD, &inputs, out, null);
    return out;
}

pub fn sub(program: *Program, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try program.arena.makeTensor(a.dtype, a.shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try program.append(&OPS.SUB, &inputs, out, null);
    return out;
}

pub fn mul(program: *Program, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try program.arena.makeTensor(a.dtype, a.shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try program.append(&OPS.MUL, &inputs, out, null);
    return out;
}

pub fn div(program: *Program, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try program.arena.makeTensor(a.dtype, a.shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try program.append(&OPS.DIV, &inputs, out, null);
    return out;
}

pub fn matmul(program: *Program, a: *const Tensor, b: *const Tensor) !*const Tensor {
    // only 2D tensors are supported for now
    if (a.shape.n_dimensions != 2 or b.shape.n_dimensions != 2) return error.ShapeMismatch;
    if (a.shape.dimensions[1] != b.shape.dimensions[0]) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const M = a.shape.dimensions[0];
    const K = b.shape.dimensions[1];

    const out = try program.arena.makeTensor(a.dtype, .fromSlice(&.{ M, K }), a.requires_grad or b.requires_grad);
    try program.append(&OPS.MATMUL, &.{ a, b }, out, null);
    return out;
}

// ============================== Tests =====================================

const testing = std.testing;

test "op: add forward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try program.createInput("a", .float32, shape, false);
    const b = try program.createInput("b", .float32, shape, false);

    const c = try add(&program, a, b);
    try program.registerOutput("c", c);

    try program.finalize(false);
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32) orelse return error.NullSlice;
    const bSlice = b.slice(f32) orelse return error.NullSlice;
    aSlice[0] = 2.0;
    aSlice[1] = 3.0;
    bSlice[0] = 4.0;
    bSlice[1] = 5.0;

    try program.forward();

    const cSlice = c.slice(f32) orelse return error.NullSlice;
    try testing.expectEqual(6.0, cSlice[0]);
    try testing.expectEqual(8.0, cSlice[1]);
}

test "op: add backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try program.createInput("a", .float32, shape, true);
    const b = try program.createInput("b", .float32, shape, true);

    const c = try add(&program, a, b);
    try program.registerOutput("c", c);

    try program.finalize(true);
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32) orelse return error.NullSlice;
    const bSlice = b.slice(f32) orelse return error.NullSlice;
    aSlice[0] = 2.0;
    aSlice[1] = 3.0;
    bSlice[0] = 4.0;
    bSlice[1] = 5.0;

    try program.forward();

    const cGradSlice = c.grad.?.slice(f32) orelse return error.NullSlice;
    cGradSlice[0] = 1.0;
    cGradSlice[1] = 1.0;
    @memset(a.grad.?.slice(f32) orelse return error.NullSlice, 0);
    @memset(b.grad.?.slice(f32) orelse return error.NullSlice, 0);

    try program.backward();

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

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try program.createInput("a", .float32, shape, false);
    const b = try program.createInput("b", .float32, shape, false);

    const c = try sub(&program, a, b);
    try program.registerOutput("c", c);

    try program.finalize(false);
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32) orelse return error.NullSlice;
    const bSlice = b.slice(f32) orelse return error.NullSlice;
    aSlice[0] = 5.0;
    aSlice[1] = 7.0;
    bSlice[0] = 2.0;
    bSlice[1] = 3.0;

    try program.forward();

    const cSlice = c.slice(f32).?;
    try testing.expectEqual(3.0, cSlice[0]);
    try testing.expectEqual(4.0, cSlice[1]);
}

test "op: sub backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try program.createInput("a", .float32, shape, true);
    const b = try program.createInput("b", .float32, shape, true);

    const c = try sub(&program, a, b);
    try program.registerOutput("c", c);

    try program.finalize(true);
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    aSlice[0] = 5.0;
    aSlice[1] = 7.0;
    bSlice[0] = 2.0;
    bSlice[1] = 3.0;

    try program.forward();

    const cGradSlice = c.grad.?.slice(f32).?;
    cGradSlice[0] = 1.0;
    cGradSlice[1] = 1.0;
    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try program.backward();

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

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try program.createInput("a", .float32, shape, false);
    const b = try program.createInput("b", .float32, shape, false);

    const c = try mul(&program, a, b);
    try program.registerOutput("c", c);

    try program.finalize(false);
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    aSlice[0] = 2.0;
    aSlice[1] = 3.0;
    bSlice[0] = 4.0;
    bSlice[1] = 5.0;

    try program.forward();

    const cSlice = c.slice(f32).?;
    try testing.expectEqual(8.0, cSlice[0]);
    try testing.expectEqual(15.0, cSlice[1]);
}

test "op: mul backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try program.createInput("a", .float32, shape, true);
    const b = try program.createInput("b", .float32, shape, true);

    const c = try mul(&program, a, b);
    try program.registerOutput("c", c);

    try program.finalize(true);
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    aSlice[0] = 2.0;
    aSlice[1] = 3.0;
    bSlice[0] = 4.0;
    bSlice[1] = 5.0;

    try program.forward();

    const cGradSlice = c.grad.?.slice(f32).?;
    cGradSlice[0] = 1.0;
    cGradSlice[1] = 1.0;
    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try program.backward();

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

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try program.createInput("a", .float32, shape, false);
    const b = try program.createInput("b", .float32, shape, false);

    const c = try div(&program, a, b);
    try program.registerOutput("c", c);

    try program.finalize(true);
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    aSlice[0] = 8.0;
    aSlice[1] = 15.0;
    bSlice[0] = 2.0;
    bSlice[1] = 3.0;

    try program.forward();

    const cSlice = c.slice(f32).?;
    try testing.expectEqual(4.0, cSlice[0]);
    try testing.expectEqual(5.0, cSlice[1]);
}

test "op: div backward" {
    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shape = comptime Shape.fromSlice(&.{2});
    const a = try program.createInput("a", .float32, shape, true);
    const b = try program.createInput("b", .float32, shape, true);

    const c = try div(&program, a, b);
    try program.registerOutput("c", c);

    try program.finalize(true);
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    const bSlice = b.slice(f32).?;
    aSlice[0] = 8.0;
    aSlice[1] = 15.0;
    bSlice[0] = 2.0;
    bSlice[1] = 3.0;

    try program.forward();

    const cGradSlice = c.grad.?.slice(f32).?;
    cGradSlice[0] = 1.0;
    cGradSlice[1] = 1.0;
    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try program.backward();

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

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shapeA: Shape = comptime .fromSlice(&.{ 2, 3 });
    const a = try program.createInput("a", .float32, shapeA, false);

    const shapeB: Shape = comptime .fromSlice(&.{ 3, 2 });
    const b = try program.createInput("b", .float32, shapeB, false);

    // (2,3) * (3,2) gives a (2,2) matrix
    const c = try matmul(&program, a, b);
    try testing.expectEqual(Shape.fromSlice(&.{ 2, 2 }), c.shape);

    try program.registerOutput("c", c);

    try program.finalize(false);
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

    try program.forward();

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

    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    const shapeA: Shape = comptime .fromSlice(&.{ 1, 2 });
    const a = try program.createInput("a", .float32, shapeA, true);

    const shapeB: Shape = comptime .fromSlice(&.{ 2, 1 });
    const b = try program.createInput("b", .float32, shapeB, true);

    // (1,2) * (2,1) gives (1,1), a dot product basically
    const c = try matmul(&program, a, b);
    try program.registerOutput("c", c);

    try program.finalize(true);
    try tensorArena.allocateStorage();

    const aSlice = a.slice(f32).?;
    @memcpy(aSlice, &[_]f32{ 1.0, 2.0 });

    const bSlice = b.slice(f32).?;
    @memcpy(bSlice, &[_]f32{ 3.0, 4.0 });

    try program.forward();

    // 3 + 2 * 4 = 11
    const cSlice = c.slice(f32).?;
    try testing.expectEqual(11.0, cSlice[0]);

    // make gradient dC = 1.0 just for simplicity
    const cGradSlice = c.grad.?.slice(f32).?;
    cGradSlice[0] = 1.0;

    @memset(a.grad.?.slice(f32).?, 0);
    @memset(b.grad.?.slice(f32).?, 0);

    try program.backward();

    // dA = dC * B^T = 1.0 * [[3,4]] = [[3,4]]
    const aGradSlice = a.grad.?.slice(f32).?;
    try testing.expectEqual(3.0, aGradSlice[0]);
    try testing.expectEqual(4.0, aGradSlice[1]);

    // dB = A^T * dC = [[1],[2]] * 1.0 = [[1],[2]]
    const bGradSlice = b.grad.?.slice(f32).?;
    try testing.expectEqual(1.0, bGradSlice[0]);
    try testing.expectEqual(2.0, bGradSlice[1]);
}
