const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// A list of available operations.
pub const Op = enum {
    Add,
    Sub,
    Div,
    Mul,
    MulScalar,
    MatMul,
    Exp,
    Log,
    Sum,
    SumAxis,
};

/// Returns the shape of the tensor resulting from the given operation.
/// Returns an error if the two operand dimensions aren't compatible with each other for the specified operation.
pub fn opShape(comptime op: Op, shape1: struct { usize, usize, usize }, shape2: anytype) error{ IncompatibleShapes, RequiresTwoShapes, InvalidAxis }!struct { usize, usize, usize } {
    switch (op) {
        inline .Add, .Sub, .Div, .Mul => {
            return switch (@typeInfo(@TypeOf(shape2))) {
                .Struct => try broadcastShape(shape1, shape2),
                else => @compileError("Expected a shape for shape2."),
            };
        },
        inline .MatMul => { //requires mat1.n_cols == mat2.n_rows
            const shape_2 = switch (@typeInfo(@TypeOf(shape2))) {
                .Struct => shape2,
                else => @compileError("Expected a shape for shape2."),
            };

            var final_shape: struct { usize, usize, usize } = .{ 0, 0, 0 };

            //requires mat1.n_cols == mat2.n_rows
            if (shape1[2] != shape_2[1])
                return error.IncompatibleShapes;

            final_shape[1] = shape1[1];
            final_shape[2] = shape_2[2];

            const batch1 = shape1[0];
            const batch2 = shape_2[0];

            if (batch1 != batch2 and batch1 != 1 and batch1 != 0 and batch2 != 1 and batch2 != 0)
                return error.IncompatibleShapes;

            final_shape[0] = @max(batch1, batch2);

            return final_shape;
        },
        inline .Exp, .Log, .MulScalar => return shape1, //same shape as the input
        inline .Sum => .{ 1, 1, 1 }, // outputs a scalar
        inline .SumAxis => {
            const axis_idx = switch (@typeInfo(@TypeOf(shape2))) {
                .ComptimeInt => shape2,
                else => @compileError("Expected an integer for shape2."),
            };

            if (axis_idx > 3)
                return error.InvalidAxis;

            var final_shape = shape1;
            final_shape[axis_idx] = 0;

            return final_shape;
        },
    }
}

/// Attempts to broadcast two shapes and returns the broadcasted shape.
/// Returns an error if the shapes aren't broadcastable.
pub fn broadcastShape(shape1: struct { usize, usize, usize }, shape2: struct { usize, usize, usize }) error{IncompatibleShapes}!struct { usize, usize, usize } {
    var final_shape: struct { usize, usize, usize } = .{ 0, 0, 0 };

    comptime var i = 2;

    inline while (i >= 0) : (i -= 1) {
        const dim1 = shape1[i];
        const dim2 = shape2[i];

        if (dim1 != dim2 and dim1 != 1 and dim1 != 0 and dim2 != 1 and dim2 != 0)
            return error.IncompatibleShapes;

        final_shape[i] = @max(dim1, dim2);
    }

    return final_shape;
}

/// Performs various shape and stride checks on the operand and result tensors to select the most adapted operation implementation.
/// - If the mat1 or mat2 or result tensors have different shapes, use the broadcasting impl (slowest)
/// - If all shapes are equal and tensors are contiguous, use the SIMD impl (fatest)
/// - else use the scalar fallback version (slower)
inline fn selectOpBackend(comptime ty: type, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty)) enum { Broadcast, SameShapeNonContiguous, SameShapeContiguous } {
    inline for (0..3) |i| {
        if (mat1.shape[i] != mat2.shape[i] or mat2.shape[i] != result.shape[i])
            return .Broadcast;
    }

    if (mat1.isContiguous() and mat2.isContiguous() and result.isContiguous()) {
        return .SameShapeContiguous;
    } else {
        return .SameShapeNonContiguous;
    }
}

/// Performs a matrix multiplication between two tensors.
/// Supports broadcasting to a common batch dimension.
/// Requires the two rightmost dimensions to be the number of columns and number of rows.
pub fn matMul(comptime ty: type, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty)) void {
    std.debug.assert(mat1.shape[2] == mat2.shape[1]);
    std.debug.assert(result.shape[1] == mat1.shape[1] and result.shape[2] == mat2.shape[2]);
    std.debug.assert(result.shape[0] == @max(mat1.shape[0], mat2.shape[0]));

    for (0..@max(result.shape[0], 1)) |b| {
        for (0..@max(result.shape[1], 1)) |i| {
            for (0..@max(result.shape[2], 1)) |j| {
                var s: ty = 0;

                for (0..@max(mat1.shape[2], 1)) |k|
                    s += mat1.get(.{ b % @max(mat1.shape[0], 1), i, k }) * mat2.get(.{ b % @max(mat2.shape[0], 1), k, j });

                result.set(.{ b, i, j }, s);
            }
        }
    }
}

/// Performs a scalar to tensor multiplication.
pub fn mulScalar(comptime ty: type, mat1: *const Tensor(ty), scalar: ty, result: *Tensor(ty)) void {
    inline for (0..3) |i|
        std.debug.assert(result.shape[i] == mat1.shape[i]);

    for (mat1.constSlice(), result.slice()) |v, *r|
        r.* = v * scalar;
}

/// Performs substraction of two tensors.
/// Supports broadcasting to a common shape.
pub fn sub(comptime ty: type, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty)) void {
    const outShape = broadcastShape(mat1.shape, mat2.shape) catch unreachable;
    inline for (0..3) |i|
        std.debug.assert(result.shape[i] == outShape[i]);

    switch (selectOpBackend(ty, mat1, mat2, result)) {
        .Broadcast => {
            for (0..@max(result.shape[0], 1)) |i| {
                for (0..@max(result.shape[1], 1)) |j| {
                    for (0..@max(result.shape[2], 1)) |k| {
                        const a = mat1.get(.{ i % @max(mat1.shape[0], 1), j % @max(mat1.shape[1], 1), k % @max(mat1.shape[2], 1) });
                        const b = mat2.get(.{ i % @max(mat2.shape[0], 1), j % @max(mat2.shape[1], 1), k % @max(mat2.shape[2], 1) });
                        result.set(.{ i, j, k }, a - b);
                    }
                }
            }
        },
        .SameShapeContiguous => {
            const vectorSize = 8;

            const arg_1 = mat1.constSlice();
            const arg_2 = mat2.constSlice();
            const res = result.slice();

            const maxVecIndex = (res.len / vectorSize) * vectorSize;
            const remIndex = res.len % vectorSize;

            var pos: usize = 0;
            while (pos < maxVecIndex) : (pos += vectorSize) {
                const vec_1: @Vector(vectorSize, ty) = arg_1[pos..][0..vectorSize].*;
                const vec_2: @Vector(vectorSize, ty) = arg_2[pos..][0..vectorSize].*;
                const res_vec: @Vector(vectorSize, ty) = vec_1 - vec_2;

                res[pos..][0..vectorSize].* = res_vec;
            }

            // processing the remaining elements which can't be vectorized.
            for (maxVecIndex..(maxVecIndex + remIndex)) |i|
                res[i] = arg_1[i] - arg_2[i];
        },
        .SameShapeNonContiguous => {
            for (0..@max(result.shape[0], 1)) |i| {
                for (0..@max(result.shape[1], 1)) |j| {
                    for (0..@max(result.shape[2], 1)) |k| {
                        const a = mat1.get(.{ i, j, k });
                        const b = mat2.get(.{ i, j, k });
                        result.set(.{ i, j, k }, a - b);
                    }
                }
            }
        },
    }
}

/// Performs element-wise multiplication of two tensors (aka the hadamard product).
/// Supports broadcasting to a common shape.
pub fn mul(comptime ty: type, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty)) void {
    const outShape = broadcastShape(mat1.shape, mat2.shape) catch unreachable;
    inline for (0..3) |i|
        std.debug.assert(result.shape[i] == outShape[i]);

    switch (selectOpBackend(ty, mat1, mat2, result)) {
        .Broadcast => {
            for (0..@max(result.shape[0], 1)) |i| {
                for (0..@max(result.shape[1], 1)) |j| {
                    for (0..@max(result.shape[2], 1)) |k| {
                        const a = mat1.get(.{ i % @max(mat1.shape[0], 1), j % @max(mat1.shape[1], 1), k % @max(mat1.shape[2], 1) });
                        const b = mat2.get(.{ i % @max(mat2.shape[0], 1), j % @max(mat2.shape[1], 1), k % @max(mat2.shape[2], 1) });
                        result.set(.{ i, j, k }, a * b);
                    }
                }
            }
        },
        .SameShapeContiguous => {
            const vectorSize = 8;

            const arg_1 = mat1.constSlice();
            const arg_2 = mat2.constSlice();
            const res = result.slice();

            const maxVecIndex = (res.len / vectorSize) * vectorSize;
            const remIndex = res.len % vectorSize;

            var pos: usize = 0;
            while (pos < maxVecIndex) : (pos += vectorSize) {
                const vec_1: @Vector(vectorSize, ty) = arg_1[pos..][0..vectorSize].*;
                const vec_2: @Vector(vectorSize, ty) = arg_2[pos..][0..vectorSize].*;
                const res_vec: @Vector(vectorSize, ty) = vec_1 * vec_2;

                res[pos..][0..vectorSize].* = res_vec;
            }

            // processing the remaining elements which can't be vectorized.
            for (maxVecIndex..(maxVecIndex + remIndex)) |i|
                res[i] = arg_1[i] * arg_2[i];
        },
        .SameShapeNonContiguous => {
            for (0..@max(result.shape[0], 1)) |i| {
                for (0..@max(result.shape[1], 1)) |j| {
                    for (0..@max(result.shape[2], 1)) |k| {
                        const a = mat1.get(.{ i, j, k });
                        const b = mat2.get(.{ i, j, k });
                        result.set(.{ i, j, k }, a * b);
                    }
                }
            }
        },
    }
}

/// Performs the addition of two tensors
/// Supports broadcasting to a common shape.
pub fn add(comptime ty: type, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty)) void {
    const outShape = broadcastShape(mat1.shape, mat2.shape) catch unreachable;
    inline for (0..3) |i|
        std.debug.assert(result.shape[i] == outShape[i]);

    switch (selectOpBackend(ty, mat1, mat2, result)) {
        .Broadcast => {
            for (0..@max(result.shape[0], 1)) |i| {
                for (0..@max(result.shape[1], 1)) |j| {
                    for (0..@max(result.shape[2], 1)) |k| {
                        const a = mat1.get(.{ i % @max(mat1.shape[0], 1), j % @max(mat1.shape[1], 1), k % @max(mat1.shape[2], 1) });
                        const b = mat2.get(.{ i % @max(mat2.shape[0], 1), j % @max(mat2.shape[1], 1), k % @max(mat2.shape[2], 1) });
                        result.set(.{ i, j, k }, a + b);
                    }
                }
            }
        },
        .SameShapeContiguous => {
            const vectorSize = 8;

            const arg_1 = mat1.constSlice();
            const arg_2 = mat2.constSlice();
            const res = result.slice();

            const maxVecIndex = (res.len / vectorSize) * vectorSize;
            const remIndex = res.len % vectorSize;

            var pos: usize = 0;
            while (pos < maxVecIndex) : (pos += vectorSize) {
                const vec_1: @Vector(vectorSize, ty) = arg_1[pos..][0..vectorSize].*;
                const vec_2: @Vector(vectorSize, ty) = arg_2[pos..][0..vectorSize].*;
                const res_vec: @Vector(vectorSize, ty) = vec_1 + vec_2;

                res[pos..][0..vectorSize].* = res_vec;
            }

            // processing the remaining elements which can't be vectorized.
            for (maxVecIndex..(maxVecIndex + remIndex)) |i|
                res[i] = arg_1[i] + arg_2[i];
        },
        .SameShapeNonContiguous => {
            for (0..@max(result.shape[0], 1)) |i| {
                for (0..@max(result.shape[1], 1)) |j| {
                    for (0..@max(result.shape[2], 1)) |k| {
                        const a = mat1.get(.{ i, j, k });
                        const b = mat2.get(.{ i, j, k });
                        result.set(.{ i, j, k }, a + b);
                    }
                }
            }
        },
    }
}

/// Performs exponentiation on the specified tensor.
pub fn exp(comptime ty: type, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    inline for (0..3) |i|
        std.debug.assert(result.shape[i] == mat1.shape[i]);

    for (mat1.constSlice(), result.slice()) |v, *r|
        r.* = std.math.exp(v);
}

/// Performs log()
pub fn log(comptime ty: type, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    inline for (0..3) |i|
        std.debug.assert(result.shape[i] == mat1.shape[i]);

    for (mat1.constSlice(), result.constSlice()) |v, *r|
        r.* = @log(v);
}

/// Sums the values of the tensor.
pub fn sum(comptime ty: type, mat1: *const Tensor(ty)) ty {
    var summed: ty = 0;
    for (mat1.constSlice()) |v|
        summed += v;

    return summed;
}

pub const ReduceOp = enum {
    Sum,
    Product,
};

/// Performs a reduction along the specified axis on the operand tensor.
pub fn reduce(comptime ty: type, comptime op: ReduceOp, mat1: *const Tensor(ty), comptime axis: usize, result: *Tensor(ty)) void {
    const axes: struct { usize, usize, usize } = switch (axis) {
        0 => .{ 1, 2, 0 },
        1 => .{ 0, 2, 1 },
        2 => .{ 0, 1, 2 },
        else => unreachable,
    };

    for (0..@max(result.shape[axes[0]], 1)) |i| {
        for (0..@max(result.shape[axes[1]], 1)) |j| {
            var s: ty = 0;
            for (0..@max(mat1.shape[axes[2]], 1)) |k| {
                var index: struct { usize, usize, usize } = .{ 0, 0, 0 };
                index[axes[0]] = i;
                index[axes[1]] = j;
                index[axes[2]] = k;

                switch (op) {
                    inline .Sum => s += mat1.get(index),
                    inline .Product => s *= mat1.get(index),
                }
            }

            var a: struct { usize, usize, usize } = .{ 0, 0, 0 };
            a[axes[0]] = i;
            a[axes[1]] = j;
            a[axes[2]] = 0;

            result.set(a, s);
        }
    }
}

test "shape broadcasting" {
    try std.testing.expectEqual(.{ 3, 2, 3 }, try broadcastShape(.{ 3, 2, 1 }, .{ 1, 2, 3 }));
    try std.testing.expectEqual(.{ 0, 2, 3 }, try broadcastShape(.{ 0, 2, 1 }, .{ 0, 1, 3 }));
    try std.testing.expectEqual(.{ 1, 2, 3 }, try broadcastShape(.{ 0, 1, 0 }, .{ 1, 2, 3 }));
    try std.testing.expectError(error.IncompatibleShapes, broadcastShape(.{ 4, 2, 9 }, .{ 1, 2, 3 }));

    // testing matrix multiplication batch dim broadcasting
    try std.testing.expectEqual(.{ 0, 2, 1 }, try opShape(.MatMul, .{ 0, 2, 3 }, .{ 0, 3, 1 }));
    try std.testing.expectEqual(.{ 6, 2, 1 }, try opShape(.MatMul, .{ 6, 2, 3 }, .{ 6, 3, 1 }));
}

test "add op test" {
    var mat1 = try Tensor(f32).empty(.{ 0, 2, 2 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    var mat2 = try Tensor(f32).empty(.{ 0, 0, 2 }, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);

    const mat3Sh = try broadcastShape(mat1.shape, mat2.shape);

    var mat3 = try Tensor(f32).empty(mat3Sh, std.testing.allocator);
    defer mat3.deinit(std.testing.allocator);

    mat1.setData(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    mat2.setData(&[_]f32{ 2.0, 2.0 });

    add(f32, &mat1, &mat2, &mat3);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3.0, 4.0, 5.0, 6.0 }, mat3.constSlice());
}

test "sub op test" {
    var mat1 = try Tensor(f32).empty(.{ 0, 2, 2 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    var mat2 = try Tensor(f32).empty(.{ 0, 0, 2 }, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);

    const mat3Sh = try broadcastShape(mat1.shape, mat2.shape);

    var mat3 = try Tensor(f32).empty(mat3Sh, std.testing.allocator);
    defer mat3.deinit(std.testing.allocator);

    mat1.setData(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    mat2.setData(&[_]f32{ 2.0, 2.0 });

    sub(f32, &mat1, &mat2, &mat3);
    try std.testing.expectEqualSlices(f32, &[_]f32{ -1.0, 0.0, 1.0, 2.0 }, mat3.constSlice());
}

test "mul op test" {
    var mat1 = try Tensor(f32).empty(.{ 0, 2, 2 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    var mat2 = try Tensor(f32).empty(.{ 0, 0, 2 }, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);

    const mat3Sh = try broadcastShape(mat1.shape, mat2.shape);

    var mat3 = try Tensor(f32).empty(mat3Sh, std.testing.allocator);
    defer mat3.deinit(std.testing.allocator);

    mat1.setData(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    mat2.setData(&[_]f32{ 2.0, 2.0 });

    mul(f32, &mat1, &mat2, &mat3);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2.0, 4.0, 6.0, 8.0 }, mat3.constSlice());
}

test "mat mul broadcasting test" {
    var mat1 = try Tensor(f32).empty(.{ 3, 3, 1 }, std.testing.allocator);
    mat1.setData(&[_]f32{ 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0 });
    defer mat1.deinit(std.testing.allocator);

    try std.testing.expectEqual(2.0, mat1.get(.{ 2, 1, 0 }));

    var mat2 = try Tensor(f32).empty(.{ 0, 1, 3 }, std.testing.allocator);
    mat2.setData(&[_]f32{ 1.0, 2.0, 3.0 });
    defer mat2.deinit(std.testing.allocator);

    var mat3 = try Tensor(f32).empty(.{ 3, 3, 3 }, std.testing.allocator);
    defer mat3.deinit(std.testing.allocator);

    matMul(f32, &mat1, &mat2, &mat3);
}

test "tensor axis sum" {
    var mat1 = try Tensor(f32).empty(.{ 3, 3, 3 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);
    for (mat1.slice(), 0..) |*v, i|
        v.* = @floatFromInt(i);

    const shape = try opShape(.SumAxis, .{ 3, 3, 3 }, 0);
    try std.testing.expectEqual(.{ 0, 3, 3 }, shape);

    var result = try Tensor(f32).empty(.{ 0, 3, 3 }, std.testing.allocator);
    defer result.deinit(std.testing.allocator);

    reduce(f32, &mat1, 0, &result); // summing along the batch size
    try std.testing.expectEqualSlices(f32, &[_]f32{ 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 48.0, 51.0 }, result.constSlice());
}
