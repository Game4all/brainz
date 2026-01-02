const tensor = @import("../tensor.zig");

const Dtype = tensor.Dtype;
const Tensor = tensor.Tensor;

//TODO: with contiguous storage, this could be SIMD-accelerated pretty easily
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

//TODO: with contiguous storage, this could be SIMD-accelerated pretty easily
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

            // the computation of some gradients require having both inputs at hand
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
// Return the gradients of both inputs
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

// Add operation

pub fn forwardAdd(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, addOp);
}

pub fn backwardAdd(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, addGradOp);
}

// Sub operation
pub fn forwardSub(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, subOp);
}

pub fn backwardSub(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, subGradOp);
}

// Mul operation
pub fn forwardMul(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, mulOp);
}

pub fn backwardMul(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, mulGradOp);
}

// Div operation
pub fn forwardDiv(inputs: []const *const Tensor, output: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseForward(inputs, output, divOp);
}

pub fn backwardDiv(inputs: []const *const Tensor, output: *const Tensor, gradOutput: *const Tensor, extraData: ?*anyopaque) !void {
    _ = extraData;
    try elementWiseBackward(inputs, output, gradOutput, divGradOp);
}
