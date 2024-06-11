const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

pub const BinaryOp = enum {
    Add,
    Sub,
    Div,
    Mul,
    Hadamard,
};

/// Returns the shape of the matrix resulting from the given operation.
/// Returns an error if the two operand dimensions aren't compatible with each other for the specified operation.
pub fn opResultShape(comptime op: BinaryOp, shape1: struct { usize, usize }, shape2: struct { usize, usize }) error{IncompatibleMatShapes}!struct { usize, usize } {
    switch (op) {
        inline .Add, .Sub, .Div, .Hadamard => { // requires both dimensions to be the same
            inline for (0..2) |i| {
                if (shape1[i] != shape2[i])
                    return error.IncompatibleMatShapes;
            }
            return shape1;
        },
        inline .Mul => { //requires mat1.n_cols == mat2.n_rows
            if (shape1[1] != shape2[0])
                return error.IncompatibleMatShapes;

            return .{ shape1[0], shape2[1] };
        },
    }
}

/// Performs matrix multiplication between two matrices.
pub fn mul(comptime ty: type, mat1: *const Matrix(ty), mat2: *const Matrix(ty), result: *Matrix(ty)) void {
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
pub fn hadamard(comptime ty: type, mat1: *const Matrix(ty), mat2: *const Matrix(ty), result: *Matrix(ty)) void {
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
    for (mat1.storage.get_slice(), result.storage.get_mut_slice()) |v, *r|
        r.* = std.math.exp(v);
}

/// Performs log()
pub fn log(comptime ty: type, mat1: *const Matrix(ty), result: *Matrix(ty)) void {
    std.debug.assert(result.shape[0] == mat1.shape[0] and result.shape[1] == mat1.shape[1]);
    for (mat1.storage.get_slice(), result.storage.get_slice()) |v, *r|
        r.* = @log(v);
}

/// Sums the values of the matrix.
pub fn sum(comptime ty: type, mat1: *const Matrix(ty)) ty {
    var summed: ty = 0;
    for (mat1.storage.get_slice()) |v|
        summed += v;

    return summed;
}

test "matrix op shape checking" {
    const shape_A = .{ 3, 1 };
    const shape_B = .{ 3, 1 };
    const shape_C = .{ 1, 3 };

    _ = try opResultShape(.Add, shape_A, shape_B);
    _ = try opResultShape(.Sub, shape_A, shape_B);
    _ = try opResultShape(.Div, shape_A, shape_B);
    _ = try opResultShape(.Hadamard, shape_A, shape_B);

    try std.testing.expectError(error.IncompatibleMatShapes, opResultShape(.Mul, shape_A, shape_B));
    try std.testing.expectEqual(.{ 3, 3 }, opResultShape(.Mul, shape_A, shape_C));
}