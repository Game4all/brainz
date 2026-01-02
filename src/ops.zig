const std = @import("std");
const tensor = @import("tensor.zig");
const prog = @import("program.zig");

const Tensor = tensor.Tensor;
const TensorArena = tensor.TensorArena;
const Dtype = tensor.Dtype;
const Shape = tensor.Shape;
const OpInfo = prog.OpInfo;
const Program = prog.Program;

const OPS = struct {
    pub const ADD: OpInfo = .{
        .name = "Add",
        .forward = forwardAdd,
        .backward = backwardAdd,
    };

    pub const SUB: OpInfo = .{
        .name = "Sub",
        .forward = forwardSub,
        .backward = backwardSub,
    };

    pub const MUL: OpInfo = .{
        .name = "Mul",
        .forward = forwardMul,
        .backward = backwardMul,
    };

    pub const DIV: OpInfo = .{
        .name = "Div",
        .forward = forwardDiv,
        .backward = backwardDiv,
    };
};

/// Performs a generic, element-wise operation on two tensors writing the result to an output tensor.
fn elementWiseForward(inputs: []const *const Tensor, output: *const Tensor, comptime op: anytype) !void {
    const a = inputs[0];
    const b = inputs[1];

    if (!output.dtype.isFloatingPoint()) return error.UnsupportedDtype; // int types do not support element-wise operations for now

    switch (output.dtype) {
        .float32 => {
            const outSlice = output.slice(f32).?;
            const aSlice = a.slice(f32).?;
            const bSlice = b.slice(f32).?;
            for (outSlice, 0..) |*v, i| {
                v.* = op(aSlice[i], bSlice[i]);
            }
        },
        .float64 => {
            const outSlice = output.slice(f64).?;
            const aSlice = a.slice(f64).?;
            const bSlice = b.slice(f64).?;
            for (outSlice, 0..) |*v, i| {
                v.* = op(aSlice[i], bSlice[i]);
            }
        },
        else => {},
    }
}

/// Performs a generic, element-wise backward pass writing the results to the gradients of the input tensors.
fn elementWiseBackward(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, comptime gradOp: anytype) !void {
    _ = output;
    const a = inputs[0];
    const b = inputs[1];

    if (!gradOutput.dtype.isFloatingPoint()) return error.UnsupportedDtype; // int types do not support element-wise operations for now

    switch (gradOutput.dtype) {
        .float32 => {
            const gradOutSlice = gradOutput.slice(f32).?;

            // the computation of some gradients require having both inputs at hand
            const aSlice = a.slice(f32).?;
            const bSlice = b.slice(f32).?;

            const aGradSlice = if (a.grad) |g| g.slice(f32) else null;
            const bGradSlice = if (b.grad) |g| g.slice(f32) else null;

            if (aGradSlice == null and bGradSlice == null) return;

            for (gradOutSlice, 0..) |gCommon, i| {
                const grads = gradOp(gCommon, aSlice[i], bSlice[i]);
                // grads is a struct { da: T, db: T }

                if (aGradSlice) |gs| gs[i] += grads.da;
                if (bGradSlice) |gs| gs[i] += grads.db;
            }
        },
        .float64 => {
            const gradOutSlice = gradOutput.slice(f64).?;
            const aSlice = a.slice(f64).?;
            const bSlice = b.slice(f64).?;

            const aGradSlice = if (a.grad) |g| g.slice(f64) else null;
            const bGradSlice = if (b.grad) |g| g.slice(f64) else null;

            if (aGradSlice == null and bGradSlice == null) return;

            for (gradOutSlice, 0..) |gCommon, i| {
                const grads = gradOp(gCommon, aSlice[i], bSlice[i]);

                if (aGradSlice) |gs| gs[i] += grads.da;
                if (bGradSlice) |gs| gs[i] += grads.db;
            }
        },
        else => {},
    }
}

// Op functions
inline fn addOp(a: anytype, b: anytype) @TypeOf(a) {
    return a + b;
}
inline fn subOp(a: anytype, b: anytype) @TypeOf(a) {
    return a - b;
}
inline fn mulOp(a: anytype, b: anytype) @TypeOf(a) {
    return a * b;
}
inline fn divOp(a: anytype, b: anytype) @TypeOf(a) {
    return a / b;
}

// Grad Op functions (Backward)
// Returns struct { da: T, db: T }
inline fn addGradOp(gradOut: anytype, a: anytype, b: anytype) struct { da: @TypeOf(gradOut), db: @TypeOf(gradOut) } {
    _ = a;
    _ = b;
    return .{ .da = gradOut, .db = gradOut };
}

inline fn subGradOp(gradOut: anytype, a: anytype, b: anytype) struct { da: @TypeOf(gradOut), db: @TypeOf(gradOut) } {
    _ = a;
    _ = b;
    return .{ .da = gradOut, .db = -gradOut };
}

inline fn mulGradOp(gradOut: anytype, a: anytype, b: anytype) struct { da: @TypeOf(gradOut), db: @TypeOf(gradOut) } {
    return .{ .da = gradOut * b, .db = gradOut * a };
}

inline fn divGradOp(gradOut: anytype, a: anytype, b: anytype) struct { da: @TypeOf(gradOut), db: @TypeOf(gradOut) } {
    const invB = 1.0 / b;
    return .{ .da = gradOut * invB, .db = gradOut * (-a * (invB * invB)) };
}

// --- Add ---
fn forwardAdd(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, addOp);
}

fn backwardAdd(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, addGradOp);
}

pub fn add(program: *Program, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try program.arena.makeTensor(a.dtype, a.shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try program.append(&OPS.ADD, &inputs, out, null);
    return out;
}

// --- Sub ---
fn forwardSub(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, subOp);
}

fn backwardSub(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, subGradOp);
}

pub fn sub(program: *Program, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try program.arena.makeTensor(a.dtype, a.shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try program.append(&OPS.SUB, &inputs, out, null);
    return out;
}

// --- Mul ---
fn forwardMul(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, mulOp);
}

fn backwardMul(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, mulGradOp);
}

pub fn mul(program: *Program, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try program.arena.makeTensor(a.dtype, a.shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try program.append(&OPS.MUL, &inputs, out, null);
    return out;
}

// --- Div ---
fn forwardDiv(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, divOp);
}

fn backwardDiv(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, divGradOp);
}

pub fn div(program: *Program, a: *const Tensor, b: *const Tensor) !*const Tensor {
    if (!a.shape.eql(b.shape)) return error.ShapeMismatch;
    if (a.dtype != b.dtype) return error.DtypeMismatch;

    const out = try program.arena.makeTensor(a.dtype, a.shape, a.requires_grad or b.requires_grad);
    const inputs = [_]*const Tensor{ a, b };
    try program.append(&OPS.DIV, &inputs, out, null);
    return out;
}

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
    // d(a/b)/db = -a/b^2 = -8/4 = -2
    try testing.expectEqual(-2.0, bGradSlice[0]);
}
