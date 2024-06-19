const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

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
};

/// Returns the shape of the matrix resulting from the given operation.
/// Returns an error if the two operand dimensions aren't compatible with each other for the specified operation.
pub fn opShape(comptime op: Op, shape1: struct { usize, usize }, shape2: ?struct { usize, usize }) error{ IncompatibleMatShapes, RequiresTwoShapes }!struct { usize, usize } {
    switch (op) {
        inline .Add, .Sub, .Div, .Mul => { // requires both dimensions to be the same
            const shape_2 = shape2 orelse return error.RequiresTwoShapes;

            inline for (0..2) |i| {
                if (shape1[i] != shape_2[i])
                    return error.IncompatibleMatShapes;
            }

            return shape1;
        },
        inline .MatMul => { //requires mat1.n_cols == mat2.n_rows
            const shape_2 = shape2 orelse return error.RequiresTwoShapes;

            if (shape1[1] != shape_2[0])
                return error.IncompatibleMatShapes;

            return .{ shape1[0], shape_2[1] };
        },
        inline .Exp, .Log, .MulScalar => return shape1, //same shape as the input
        inline .Sum => .{ 1, 1 }, // outputs a scalar
    }
}

/// Performs matrix multiplication between two matrices.
pub fn matMul(comptime ty: type, mat1: *const Matrix(ty), mat2: *const Matrix(ty), result: *Matrix(ty)) void {
    std.debug.assert(mat1.shape[1] == mat2.shape[0]);
    std.debug.assert(result.shape[0] == mat1.shape[0] and result.shape[1] == mat2.shape[1]);

    for (0..mat1.shape[0]) |i| {
        for (0..mat2.shape[1]) |j| {
            var s: ty = 0;
            for (0..mat1.shape[1]) |k|
                s += mat1.get(.{ i, k }) * mat2.get(.{ k, j });

            result.set(.{ i, j }, s);
        }
    }
}

/// Performs a scalar to matrix multiplication.
pub fn mulScalar(comptime ty: type, mat1: *const Matrix(ty), scalar: ty, result: *Matrix(ty)) void {
    for (mat1.constSlice(), result.slice()) |v, *r|
        r.* = v * scalar;
}

/// Performs addition of two matrices.
pub fn add(comptime ty: type, mat1: *const Matrix(ty), mat2: *const Matrix(ty), result: *Matrix(ty)) void {
    std.debug.assert(mat1.shape[0] == mat2.shape[0] and mat1.shape[1] == mat2.shape[1]);
    std.debug.assert(result.shape[0] == mat1.shape[0] and result.shape[1] == mat1.shape[1]);

    for (0..result.shape[0]) |i| {
        for (0..result.shape[1]) |j| {
            result.set(.{ i, j }, mat1.get(.{ i, j }) + mat2.get(.{ i, j }));
        }
    }
}

/// Performs substraction of two matrices.
pub fn sub(comptime ty: type, mat1: *const Matrix(ty), mat2: *const Matrix(ty), result: *Matrix(ty)) void {
    std.debug.assert(mat1.shape[0] == mat2.shape[0] and mat1.shape[1] == mat2.shape[1]);
    std.debug.assert(result.shape[0] == mat1.shape[0] and result.shape[1] == mat1.shape[1]);

    for (0..result.shape[0]) |i| {
        for (0..result.shape[1]) |j| {
            result.set(.{ i, j }, mat1.get(.{ i, j }) - mat2.get(.{ i, j }));
        }
    }
}

/// Performs the hadamard product aka element-wise multiplication of two matrices.
pub fn mul(comptime ty: type, mat1: *const Matrix(ty), mat2: *const Matrix(ty), result: *Matrix(ty)) void {
    std.debug.assert(mat1.shape[0] == mat2.shape[0] and mat1.shape[1] == mat2.shape[1]);
    std.debug.assert(result.shape[0] == mat1.shape[0] and result.shape[1] == mat1.shape[1]);

    for (0..result.shape[0]) |i| {
        for (0..result.shape[1]) |j| {
            result.set(.{ i, j }, mat1.get(.{ i, j }) * mat2.get(.{ i, j }));
        }
    }
}

/// Performs exponentiation on the specified matrix.
pub fn exp(comptime ty: type, mat1: *const Matrix(ty), result: *Matrix(ty)) void {
    std.debug.assert(result.shape[0] == mat1.shape[0] and result.shape[1] == mat1.shape[1]);
    for (mat1.constSlice(), result.slice()) |v, *r|
        r.* = std.math.exp(v);
}

/// Performs log()
pub fn log(comptime ty: type, mat1: *const Matrix(ty), result: *Matrix(ty)) void {
    std.debug.assert(result.shape[0] == mat1.shape[0] and result.shape[1] == mat1.shape[1]);
    for (mat1.constSlice(), result.constSlice()) |v, *r|
        r.* = @log(v);
}

/// Sums the values of the matrix.
pub fn sum(comptime ty: type, mat1: *const Matrix(ty)) ty {
    var summed: ty = 0;
    for (mat1.constSlice()) |v|
        summed += v;

    return summed;
}

test "matrix op shape checking" {
    const shape_A = .{ 3, 1 };
    const shape_B = .{ 3, 1 };
    const shape_C = .{ 1, 3 };

    _ = try opShape(.Add, shape_A, shape_B);
    _ = try opShape(.Sub, shape_A, shape_B);
    _ = try opShape(.Div, shape_A, shape_B);
    _ = try opShape(.Mul, shape_A, shape_B);

    try std.testing.expectError(error.IncompatibleMatShapes, opShape(.MatMul, shape_A, shape_B));
    try std.testing.expectEqual(.{ 3, 3 }, opShape(.MatMul, shape_A, shape_C));
}
