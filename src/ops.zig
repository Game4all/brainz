const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Device = @import("device/Device.zig");

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
    Reduce,
    Transpose,
    Cast,
    ArgMax,
    ElementWiseEq,

    // activations
    Sigmoid,
    SigmoidBackprop,
    ReLu,
    ReLuBackprop,
    SiLu,
    SiLuBackprop,
};

/// Returns the shape of the tensor resulting from the given operation.
/// Returns an error if the two operand dimensions aren't compatible with each other for the specified operation.
pub fn opShape(comptime op: Op, shape1: struct { usize, usize, usize }, shape2: anytype) error{ IncompatibleShapes, RequiresTwoShapes, InvalidAxis }!struct { usize, usize, usize } {
    switch (op) {
        inline .Add, .Sub, .Div, .Mul, .ElementWiseEq => {
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
        inline .Cast, .Exp, .Log, .MulScalar, .Sigmoid, .SigmoidBackprop, .ReLu, .ReLuBackprop, .SiLu, .SiLuBackprop => return shape1, //same shape as the input
        inline .Sum => .{ 1, 1, 1 }, // outputs a scalar
        inline .Reduce, .ArgMax => {
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
        inline .Transpose => return .{ shape1.@"0", shape1.@"2", shape1.@"1" },
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

/// Performs a Tensor multiplication between two tensors.
/// Supports broadcasting to a common batch dimension.
/// Requires the two rightmost dimensions to be the number of columns and number of rows.
pub fn matMul(comptime ty: type, device: Device, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty)) void {
    std.debug.assert(mat1.shape[2] == mat2.shape[1]);
    std.debug.assert(result.shape[1] == mat1.shape[1] and result.shape[2] == mat2.shape[2]);
    std.debug.assert(result.shape[0] == @max(mat1.shape[0], mat2.shape[0]));

    const dispatch_fn = (struct {
        pub fn op(arg: *const Device.Dispatch) void {
            const arg_1: *const Tensor(ty) = @alignCast(@ptrCast(arg.args[0]));
            const arg_2: *const Tensor(ty) = @alignCast(@ptrCast(arg.args[1]));
            var arg_result: *Tensor(ty) = @alignCast(@ptrCast(arg.args[2]));
            const b = arg.n_i;

            for (0..@max(arg_result.shape[2], 1)) |j| { //maybe parallelize more
                for (0..@max(arg_result.shape[1], 1)) |i| {
                    var s: ty = 0;

                    for (0..@max(arg_1.shape[2], 1)) |k|
                        s += arg_1.get(.{ b % @max(arg_1.shape[0], 1), i, k }) * arg_2.get(.{ b % @max(arg_2.shape[0], 1), k, j });

                    arg_result.set(.{ b, i, j }, s);
                }
            }
        }
    }).op;

    var work: Device.Dispatch = .{ .func = dispatch_fn };
    work.args[0] = @ptrCast(@constCast(mat1));
    work.args[1] = @ptrCast(@constCast(mat2));
    work.args[2] = @ptrCast(@constCast(result));

    device.dispatchChunks(work, @max(1, result.shape.@"0")) catch unreachable;
}

/// Performs a scalar to tensor multiplication.
pub fn mulScalar(comptime ty: type, device: Device, mat1: *const Tensor(ty), scalar: ty, result: *Tensor(ty)) void {
    std.debug.assert(std.meta.eql(mat1.shape, result.shape));

    // intermediate type used for encoding the scalar value
    const encScalarType = if (@sizeOf(ty) > @sizeOf(u32)) u64 else u32;
    const encoded_scalar_pointer: *anyopaque = @ptrFromInt(@as(encScalarType, @bitCast(scalar)));

    return opUnaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, ctx: *anyopaque, a: anytype) @Vector(vectorSize, ty) {
            const alpha: ty = @bitCast(@as(encScalarType, @intCast(@intFromPtr(ctx))));

            return @as(@Vector(vectorSize, ty), @splat(alpha)) * a;
        }

        pub inline fn scalar_func(ctx: *anyopaque, a: anytype) ty {
            const alpha: ty = @bitCast(@as(encScalarType, @intCast(@intFromPtr(ctx))));

            return alpha * a;
        }
    }, device, encoded_scalar_pointer, mat1, result);
}

/// Performs substraction of two tensors.
/// Supports broadcasting to a common shape.
pub fn sub(comptime ty: type, device: Device, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty), args: struct { alpha: ty = 1 }) void {
    const outShape = broadcastShape(mat1.shape, mat2.shape) catch unreachable;
    std.debug.assert(std.meta.eql(outShape, result.shape));

    const encScalarType = if (@sizeOf(ty) > @sizeOf(u32)) u64 else u32;
    const encoded_scalar_pointer: *anyopaque = @ptrFromInt(@as(encScalarType, @bitCast(args.alpha)));

    return opBinaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, ctx: *anyopaque, a: anytype, b: anytype) @Vector(vectorSize, ty) {
            const alpha: ty = @bitCast(@as(encScalarType, @intCast(@intFromPtr(ctx))));
            const scaler: @Vector(vectorSize, ty) = @splat(alpha);
            return a - scaler * b;
        }

        pub inline fn scalar_func(ctx: *anyopaque, a: anytype, b: anytype) ty {
            const alpha: ty = @bitCast(@as(encScalarType, @intCast(@intFromPtr(ctx))));
            return a - alpha * b;
        }
    }, device, encoded_scalar_pointer, mat1, mat2, result);
}

/// Performs element-wise multiplication of two tensors (aka the hadamard product).
/// Supports broadcasting to a common shape.
pub fn mul(comptime ty: type, device: Device, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty)) void {
    const outShape = broadcastShape(mat1.shape, mat2.shape) catch unreachable;
    std.debug.assert(std.meta.eql(outShape, result.shape));

    return opBinaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: anytype, a: anytype, b: anytype) @Vector(vectorSize, ty) {
            return a * b;
        }

        pub inline fn scalar_func(_: anytype, a: anytype, b: anytype) ty {
            return a * b;
        }
    }, device, undefined, mat1, mat2, result);
}

/// Performs the addition of two tensors
/// Supports broadcasting to a common shape.
/// Supports specifying an alpha specifier to scale the added value.
pub fn add(comptime ty: type, device: Device, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty), args: struct { alpha: ty = 1 }) void {
    const outShape = broadcastShape(mat1.shape, mat2.shape) catch unreachable;
    std.debug.assert(std.meta.eql(outShape, result.shape));

    const encScalarType = if (@sizeOf(ty) > @sizeOf(u32)) u64 else u32;
    const encoded_scalar_pointer: *anyopaque = @ptrFromInt(@as(encScalarType, @bitCast(args.alpha)));

    return opBinaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, ctx: *anyopaque, a: anytype, b: anytype) @Vector(vectorSize, ty) {
            const alpha: ty = @bitCast(@as(encScalarType, @intCast(@intFromPtr(ctx))));
            const scaler: @Vector(vectorSize, ty) = @splat(alpha);
            return a + scaler * b;
        }

        pub inline fn scalar_func(ctx: anytype, a: anytype, b: anytype) ty {
            const alpha: ty = @bitCast(@as(encScalarType, @intCast(@intFromPtr(ctx))));
            return a + alpha * b;
        }
    }, device, encoded_scalar_pointer, mat1, mat2, result);
}

/// Performs the division of a tensor by another.
/// Supports broadcasting to a common shape.
pub fn div(comptime ty: type, device: Device, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty)) void {
    const outShape = broadcastShape(mat1.shape, mat2.shape) catch unreachable;
    std.debug.assert(std.meta.eql(outShape, result.shape));

    return opBinaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: anytype, a: anytype, b: anytype) @Vector(vectorSize, ty) {
            return a / b;
        }

        pub inline fn scalar_func(_: anytype, a: anytype, b: anytype) ty {
            return a / b;
        }
    }, device, undefined, mat1, mat2, result);
}

/// Performs exponentiation on the specified tensor.
pub fn exp(comptime ty: type, device: Device, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    std.debug.assert(std.meta.eql(mat1.shape, result.shape));

    return opUnaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: anytype, a: anytype) @Vector(vectorSize, ty) {
            return @exp(a);
        }

        pub inline fn scalar_func(_: anytype, a: anytype) ty {
            return @exp(a);
        }
    }, device, undefined, mat1, result);
}

/// Performs log()
pub fn log(comptime ty: type, device: Device, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    std.debug.assert(std.meta.eql(mat1.shape, result.shape));

    return opUnaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: anytype, a: anytype) @Vector(vectorSize, ty) {
            return @log(a);
        }

        pub inline fn scalar_func(_: anytype, a: anytype) ty {
            return @log(a);
        }
    }, device, undefined, mat1, result);
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

/// Cast a tensor of one type to another.
pub fn cast(comptime in_ty: type, comptime out_ty: type, in: *const Tensor(in_ty), out: *Tensor(out_ty)) void {
    std.debug.assert(std.meta.eql(in.shape, out.shape));
    for (in.constSlice(), out.slice()) |i, *v| {
        switch (@typeInfo(in_ty)) {
            .Int => switch (@typeInfo(out_ty)) {
                .Int => v.* = @intCast(i),
                .Float => v.* = @floatFromInt(i),
                .Bool => v.* = i != 0,
                else => @compileError("Unsupported target type for tensor typecasting"),
            },
            .Float => switch (@typeInfo(out_ty)) {
                .Int => v.* = @intFromFloat(i),
                .Float => v.* = @floatCast(i),
                .Bool => v.* = i != 0,
                else => @compileError("Unsupported target type for tensor typecasting"),
            },
            .Bool => switch (@typeInfo(out_ty)) {
                .Int => v.* = @intFromBool(i),
                .Float => v.* = @floatFromInt(@intFromBool(i)),
                .Bool => v.* = i,
                else => @compileError("Unsupported target type for tensor typecasting"),
            },
            else => @compileError("Unsupported source type for tensor typecasting"),
        }
    }
}

//todo: merge this with reduce?
/// Stores the indices of the maximum values of operand tensor along the specified dimension in the result tensor.
pub fn argmax(comptime ty: type, in: *const Tensor(ty), comptime axis: usize, result: *Tensor(usize)) void {
    const axes: struct { usize, usize, usize } = switch (axis) {
        0 => .{ 1, 2, 0 },
        1 => .{ 0, 2, 1 },
        2 => .{ 0, 1, 2 },
        else => unreachable,
    };

    for (0..@max(result.shape[axes[0]], 1)) |i| {
        for (0..@max(result.shape[axes[1]], 1)) |j| {
            var maximum: ty = 0;
            var max_index: usize = 0;

            for (0..@max(in.shape[axes[2]], 1)) |k| {
                var index: struct { usize, usize, usize } = .{ 0, 0, 0 };
                index[axes[0]] = i;
                index[axes[1]] = j;
                index[axes[2]] = k;

                const val = in.get(index);
                if (val > maximum) {
                    max_index = k;
                    maximum = val;
                }
            }

            var a: struct { usize, usize, usize } = .{ 0, 0, 0 };
            a[axes[0]] = i;
            a[axes[1]] = j;
            a[axes[2]] = 0;

            result.set(a, max_index);
        }
    }
}

/// Performs element-wise equality comparision on two tensors.
/// Stores 1 or 0 in the result tensor.
pub fn elementWiseEq(comptime ty: type, device: Device, arg1: *const Tensor(ty), arg2: *const Tensor(ty), result: *Tensor(ty)) void {
    return opBinaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: anytype, a: anytype, b: anytype) @Vector(vectorSize, ty) {
            const ones: @Vector(vectorSize, ty) = @splat(1);
            const zeroes: @Vector(vectorSize, ty) = @splat(0);

            const equality: @Vector(vectorSize, bool) = a == b;

            return @select(ty, equality, ones, zeroes);
        }

        pub inline fn scalar_func(_: anytype, a: anytype, b: anytype) ty {
            if (a == b) {
                return 1;
            } else {
                return 0;
            }
        }
    }, device, undefined, arg1, arg2, result);
}

// ========== Activation functions as operations ======================

/// Sigmoid activation.
pub fn sigmoid(comptime ty: type, device: Device, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    return opUnaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: anytype, a: anytype) @Vector(vectorSize, ty) {
            const ones: @Vector(vectorSize, ty) = @splat(1);
            return ones / (ones + std.math.exp(-a));
        }

        pub inline fn scalar_func(_: anytype, a: anytype) ty {
            return 1 / (1 + std.math.exp(-a));
        }
    }, device, undefined, mat1, result);
}

/// Sigmoid activation backpropagation.
pub fn sigmoidBackprop(comptime ty: type, device: Device, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    return opUnaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: *anyopaque, a: anytype) @Vector(vectorSize, ty) {
            const ones: @Vector(vectorSize, ty) = @splat(1);
            const s = ones / (ones + std.math.exp(-a));
            return s - (s * s);
        }

        pub inline fn scalar_func(_: *anyopaque, a: anytype) ty {
            const s = 1 / (1 + std.math.exp(-a));
            return s - (s * s);
        }
    }, device, undefined, mat1, result);
}

/// ReLu activation.
pub fn relu(comptime ty: type, device: Device, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    return opUnaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: *anyopaque, a: anytype) @Vector(vectorSize, ty) {
            return @max(a, @as(@Vector(vectorSize, ty), @splat(0)));
        }

        pub inline fn scalar_func(_: *anyopaque, a: anytype) ty {
            return @max(a, 0);
        }
    }, device, undefined, mat1, result);
}

/// ReLu activation backpropagation.
pub fn reluBackprop(comptime ty: type, device: Device, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    return opUnaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: anytype, a: anytype) @Vector(vectorSize, ty) {
            return @max(std.math.sign(a), @as(@Vector(vectorSize, ty), @splat(0)));
        }

        pub inline fn scalar_func(_: anytype, a: anytype) ty {
            return @max(std.math.sign(a), 0);
        }
    }, device, undefined, mat1, result);
}

/// SiLu activation aka sigmoid linear unit.
pub fn silu(comptime ty: type, device: Device, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    return opUnaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: anytype, a: anytype) @Vector(vectorSize, ty) {
            const ones: @Vector(vectorSize, ty) = @splat(1);
            return a / (ones + std.math.exp(-a));
        }

        pub inline fn scalar_func(_: anytype, a: anytype) ty {
            return a / (1 + std.math.exp(-a));
        }
    }, device, undefined, mat1, result);
}

/// SiLu derivative.
pub fn siluBackprop(comptime ty: type, device: Device, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    return opUnaryImpl(ty, struct {
        pub inline fn simd_func(comptime vectorSize: comptime_int, _: anytype, a: anytype) @Vector(vectorSize, ty) {
            const ones: @Vector(vectorSize, ty) = @splat(1);
            const s = ones / (ones + std.math.exp(-a));

            return s * (ones + a * (ones - s));
        }

        pub inline fn scalar_func(_: anytype, a: anytype) ty {
            const s = 1 / (1 + std.math.exp(-a));
            return s * (1 + a * (1 - s));
        }
    }, device, undefined, mat1, result);
}

// ================== Unary op implementation ================================

fn opUnaryImpl(comptime ty: type, comptime op_funcs: anytype, device: Device, userptr: *anyopaque, mat1: *const Tensor(ty), result: *Tensor(ty)) void {
    const dispatch_fn = (struct {
        pub fn dispatch_fn(dispatch: *const Device.Dispatch) void {
            const arg_1: *const Tensor(ty) = @alignCast(@ptrCast(dispatch.args[0]));
            const arg_result: *Tensor(ty) = @alignCast(@ptrCast(dispatch.args[1]));
            const usrptr: *anyopaque = dispatch.args[2];
            const b = dispatch.n_i;

            opUnaryImplDispatch(ty, op_funcs, usrptr, arg_1, arg_result, b);
        }
    }).dispatch_fn;

    var dispatch: Device.Dispatch = .{ .func = dispatch_fn };
    dispatch.args[0] = @constCast(@ptrCast(mat1));
    dispatch.args[1] = @constCast(@ptrCast(result));
    dispatch.args[2] = userptr;

    device.dispatchChunks(dispatch, @max(result.shape.@"0", 1)) catch unreachable;
}

inline fn opUnaryImplDispatch(comptime ty: type, comptime op_funcs: anytype, userptr: *anyopaque, mat1: *const Tensor(ty), result: *Tensor(ty), b: usize) void {
    const arg_1 = mat1.constSlice();
    const res = result.slice();

    const batchStride = result.strides.@"0";

    var pos: usize = 0;

    if (@hasDecl(op_funcs, "simd_func")) {
        if (std.simd.suggestVectorLength(ty)) |vectorSize| {
            const maxVecIndex = (batchStride / vectorSize) * vectorSize;

            while (pos < maxVecIndex) : (pos += vectorSize) {
                const arg1_i = b * batchStride + pos;
                const vec_1: @Vector(vectorSize, ty) = arg_1[arg1_i..][0..vectorSize].*;
                const res_vec: @Vector(vectorSize, ty) = op_funcs.simd_func(vectorSize, userptr, vec_1);
                res[arg1_i..][0..vectorSize].* = res_vec;
            }

            pos = maxVecIndex;
        }
    }

    // processing the remaining elements which can't be vectorized.
    for (pos..batchStride) |i|
        res[b * batchStride + i] = op_funcs.scalar_func(userptr, arg_1[b * batchStride + i]);
}

/// Scalar operation dispatch implementation.
/// Slow.
inline fn opBinaryImplScalarDispatch(comptime ty: type, comptime op_funcs: anytype, userptr: *anyopaque, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty), i: usize) void {
    for (0..@max(result.shape[1], 1)) |j| {
        for (0..@max(result.shape[2], 1)) |k| {
            const a = mat1.get(.{ i % @max(mat1.shape[0], 1), j % @max(mat1.shape[1], 1), k % @max(mat1.shape[2], 1) });
            const b = mat2.get(.{ i % @max(mat2.shape[0], 1), j % @max(mat2.shape[1], 1), k % @max(mat2.shape[2], 1) });
            result.set(.{ i, j, k }, op_funcs.scalar_func(userptr, a, b));
        }
    }
}

/// Fast-path operation dispatch implementation.
/// Uses SIMD for speed
inline fn opBinaryImplSimdDispatch(comptime ty: type, op_funcs: anytype, userptr: *anyopaque, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty), b: usize) void {
    const arg_1 = mat1.constSlice();
    const arg_2 = mat2.constSlice();
    const res = result.slice();

    const batchStride = result.strides.@"0";

    var pos: usize = 0;

    if (std.simd.suggestVectorLength(ty)) |vectorSize| {
        const maxVecIndex = (batchStride / vectorSize) * vectorSize;

        while (pos < maxVecIndex) : (pos += vectorSize) {
            const arg1_i = (b % @max(mat1.shape.@"0", 1)) * batchStride + pos;
            const arg2_i = (b % @max(mat2.shape.@"0", 1)) * batchStride + pos;

            const vec_1: @Vector(vectorSize, ty) = arg_1[arg1_i..][0..vectorSize].*;
            const vec_2: @Vector(vectorSize, ty) = arg_2[arg2_i..][0..vectorSize].*;
            const res_vec: @Vector(vectorSize, ty) = op_funcs.simd_func(vectorSize, userptr, vec_1, vec_2);

            res[(b * batchStride + pos)..][0..vectorSize].* = res_vec;
        }

        pos = maxVecIndex;
    }

    for (pos..batchStride) |i| {
        const arg1_i = (b % @max(mat1.shape.@"0", 1)) * batchStride + i;
        const arg2_i = (b % @max(mat2.shape.@"0", 1)) * batchStride + i;
        res[b * batchStride + i] = op_funcs.scalar_func(userptr, arg_1[arg1_i], arg_2[arg2_i]);
    }
}

/// Performs various shape and stride checks on the operand and result tensors to select the most adapted operation implementation and dispatches it.
/// - If the mat1 or mat2 or result tensors have different shapes, or aren't contiguous in memory, use the broadcasting impl (slow)
/// - If all shapes are equal and tensors are contiguous, use the SIMD impl (faster)
fn opBinaryImpl(comptime ty: type, comptime op_funcs: anytype, device: Device, userpointer: *anyopaque, mat1: *const Tensor(ty), mat2: *const Tensor(ty), result: *Tensor(ty)) void {
    if (mat1.isContiguous() and mat2.isContiguous() and result.isContiguous() and canDoSimd(mat1.shape, mat2.shape) and canDoSimd(mat2.shape, result.shape)) {
        const dispatch_fn = (struct {
            pub fn dispatch_fn(arg: *const Device.Dispatch) void {
                const arg_1: *const Tensor(ty) = @alignCast(@ptrCast(arg.args[0]));
                const arg_2: *const Tensor(ty) = @alignCast(@ptrCast(arg.args[1]));
                const arg_result: *Tensor(ty) = @alignCast(@ptrCast(arg.args[2]));
                const userptr: *anyopaque = arg.args[3];
                const b = arg.n_i;

                opBinaryImplSimdDispatch(ty, op_funcs, userptr, arg_1, arg_2, arg_result, b);
            }
        }).dispatch_fn;

        var dispatch: Device.Dispatch = .{ .func = dispatch_fn };
        dispatch.args[0] = @constCast(@ptrCast(mat1));
        dispatch.args[1] = @constCast(@ptrCast(mat2));
        dispatch.args[2] = @constCast(@ptrCast(result));
        dispatch.args[3] = @constCast(@ptrCast(userpointer));

        device.dispatchChunks(dispatch, @max(result.shape.@"0", 1)) catch unreachable;
    } else {
        const dispatch_fn = (struct {
            pub fn dispatch_fn(arg: *const Device.Dispatch) void {
                const arg_1: *const Tensor(ty) = @alignCast(@ptrCast(arg.args[0]));
                const arg_2: *const Tensor(ty) = @alignCast(@ptrCast(arg.args[1]));
                const arg_result: *Tensor(ty) = @alignCast(@ptrCast(arg.args[2]));
                const userptr: *anyopaque = arg.args[3];
                const b = arg.n_i;

                opBinaryImplScalarDispatch(ty, op_funcs, userptr, arg_1, arg_2, arg_result, b);
            }
        }).dispatch_fn;

        var dispatch: Device.Dispatch = .{ .func = dispatch_fn };
        dispatch.args[0] = @constCast(@ptrCast(mat1));
        dispatch.args[1] = @constCast(@ptrCast(mat2));
        dispatch.args[2] = @constCast(@ptrCast(result));
        dispatch.args[3] = @constCast(@ptrCast(userpointer));

        device.dispatchChunks(dispatch, @max(result.shape.@"0", 1)) catch unreachable;
    }
}

// Check if two shapes have matching dimensions except for the batch dimensions
// Enables SIMD operations for batch computation if true.
inline fn canDoSimd(shape1: struct { usize, usize, usize }, shape2: struct { usize, usize, usize }) bool {
    comptime var i = 2;
    inline while (i > 0) : (i -= 1) {
        if (shape1[i] != shape2[i])
            return false;
    }
    return true;
}

test "shape broadcasting" {
    try std.testing.expectEqual(.{ 3, 2, 3 }, try broadcastShape(.{ 3, 2, 1 }, .{ 1, 2, 3 }));
    try std.testing.expectEqual(.{ 0, 2, 3 }, try broadcastShape(.{ 0, 2, 1 }, .{ 0, 1, 3 }));
    try std.testing.expectEqual(.{ 1, 2, 3 }, try broadcastShape(.{ 0, 1, 0 }, .{ 1, 2, 3 }));
    try std.testing.expectError(error.IncompatibleShapes, broadcastShape(.{ 4, 2, 9 }, .{ 1, 2, 3 }));

    // testing Tensor multiplication batch dim broadcasting
    try std.testing.expectEqual(.{ 0, 2, 1 }, try opShape(.MatMul, .{ 0, 2, 3 }, .{ 0, 3, 1 }));
    try std.testing.expectEqual(.{ 6, 2, 1 }, try opShape(.MatMul, .{ 6, 2, 3 }, .{ 6, 3, 1 }));
}

test "add op test" {
    var mat1 = try Tensor(f32).init(.{ 0, 2, 2 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    var mat2 = try Tensor(f32).init(.{ 0, 0, 2 }, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);

    const mat3Sh = try broadcastShape(mat1.shape, mat2.shape);

    var mat3 = try Tensor(f32).init(mat3Sh, std.testing.allocator);
    defer mat3.deinit(std.testing.allocator);

    mat1.setData(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    mat2.setData(&[_]f32{ 2.0, 2.0 });

    add(f32, Device.DummyDevice, &mat1, &mat2, &mat3, .{});
    try std.testing.expectEqualSlices(f32, &[_]f32{ 3.0, 4.0, 5.0, 6.0 }, mat3.constSlice());
}

test "sub op test" {
    var mat1 = try Tensor(f32).init(.{ 0, 2, 2 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    var mat2 = try Tensor(f32).init(.{ 0, 0, 2 }, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);

    const mat3Sh = try broadcastShape(mat1.shape, mat2.shape);

    var mat3 = try Tensor(f32).init(mat3Sh, std.testing.allocator);
    defer mat3.deinit(std.testing.allocator);

    mat1.setData(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    mat2.setData(&[_]f32{ 2.0, 2.0 });

    sub(f32, Device.DummyDevice, &mat1, &mat2, &mat3, .{});
    try std.testing.expectEqualSlices(f32, &[_]f32{ -1.0, 0.0, 1.0, 2.0 }, mat3.constSlice());
}

test "mul op test" {
    var mat1 = try Tensor(f32).init(.{ 0, 2, 2 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    var mat2 = try Tensor(f32).init(.{ 0, 0, 2 }, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);

    const mat3Sh = try broadcastShape(mat1.shape, mat2.shape);

    var mat3 = try Tensor(f32).init(mat3Sh, std.testing.allocator);
    defer mat3.deinit(std.testing.allocator);

    mat1.setData(&[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    mat2.setData(&[_]f32{ 2.0, 2.0 });

    mul(f32, Device.DummyDevice, &mat1, &mat2, &mat3);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 2.0, 4.0, 6.0, 8.0 }, mat3.constSlice());
}

test "mat mul broadcasting test" {
    var mat1 = try Tensor(f32).init(.{ 3, 3, 1 }, std.testing.allocator);
    mat1.setData(&[_]f32{ 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0 });
    defer mat1.deinit(std.testing.allocator);

    try std.testing.expectEqual(2.0, mat1.get(.{ 2, 1, 0 }));

    var mat2 = try Tensor(f32).init(.{ 0, 1, 3 }, std.testing.allocator);
    mat2.setData(&[_]f32{ 1.0, 2.0, 3.0 });
    defer mat2.deinit(std.testing.allocator);

    var mat3 = try Tensor(f32).init(.{ 3, 3, 3 }, std.testing.allocator);
    defer mat3.deinit(std.testing.allocator);

    matMul(f32, Device.DummyDevice, &mat1, &mat2, &mat3);
}

test "tensor axis sum" {
    var mat1 = try Tensor(f32).init(.{ 3, 3, 3 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);
    for (mat1.slice(), 0..) |*v, i|
        v.* = @floatFromInt(i);

    const shape = try opShape(.Reduce, .{ 3, 3, 3 }, 0);
    try std.testing.expectEqual(.{ 0, 3, 3 }, shape);

    var result = try Tensor(f32).init(.{ 0, 3, 3 }, std.testing.allocator);
    defer result.deinit(std.testing.allocator);

    reduce(f32, .Sum, &mat1, 0, &result); // summing along the batch size
    try std.testing.expectEqualSlices(f32, &[_]f32{ 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 48.0, 51.0 }, result.constSlice());
}

test "tensor casting" {
    var mat1 = try Tensor(f32).init(.{ 0, 0, 16 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);
    var mat2 = try Tensor(u8).init(.{ 0, 0, 16 }, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);

    for (mat1.slice(), 0..) |*value, i|
        value.* = @floatFromInt(i);

    cast(f32, u8, &mat1, &mat2);

    for (mat2.constSlice(), 0..) |value, i|
        try std.testing.expectEqual(@as(u8, @intCast(i)), value);
}

test "tensor argmax" {
    var mat1 = try Tensor(f32).init(.{ 0, 16, 1 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    for (0..mat1.shape.@"1") |i|
        mat1.set(.{ 0, i, 0 }, @floatFromInt(i));

    var res = try Tensor(usize).init(.{ 0, 0, 1 }, std.testing.allocator);
    defer res.deinit(std.testing.allocator);

    argmax(f32, &mat1, 1, &res);
    try std.testing.expectEqual(res.get(.{ 0, 0, 0 }), 15);
}

test "tensor element wise equal" {
    var mat1 = try Tensor(f32).init(.{ 0, 0, 8 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    for (0..mat1.shape.@"1") |i|
        mat1.set(.{ 0, 0, i }, @floatFromInt(i));

    var mat2 = try Tensor(f32).init(mat1.shape, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);
    @memcpy(mat2.slice(), mat1.constSlice());
    mat2.set(.{ 0, 0, 3 }, 0.0);

    var res = try Tensor(f32).init(mat1.shape, std.testing.allocator);
    defer res.deinit(std.testing.allocator);
    elementWiseEq(f32, Device.DummyDevice, &mat1, &mat2, &res);

    const result = &[_]f32{ 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0 };
    try std.testing.expectEqualSlices(f32, @constCast(result), res.constSlice());
}
