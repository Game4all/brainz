const std = @import("std");

pub const Df32 = Dfloat(f32);
pub const Df64 = Dfloat(f64);

/// A float-based dual number implementation for automatic differentiation.
pub fn Dfloat(comptime ty: type) type {
    return struct {
        value: ty,
        derivative: ty,

        pub const NUM_TYPE = ty;

        /// Creates a litteral of value which doesn't need to have its gradient calculated.
        pub inline fn literal(value: ty) @This() {
            return .{ .value = value, .derivative = 0.0 };
        }

        /// Creates value for a variable of value `val`.
        pub inline fn with_grad(val: ty) @This() {
            return .{ .value = val, .derivative = 1.0 };
        }

        pub inline fn add(self: *const @This(), other: *const @This()) @This() {
            return .{ .value = self.value + other.value, .derivative = self.derivative + other.derivative };
        }

        pub inline fn sub(self: *const @This(), other: *const @This()) @This() {
            return .{ .value = self.value - other.value, .derivative = self.derivative - other.derivative };
        }

        pub inline fn mul(self: *const @This(), other: *const @This()) @This() {
            return .{ .value = self.value * other.value, .derivative = self.value * other.derivative + self.derivative * other.value };
        }

        pub inline fn div(self: *const @This(), other: *const @This()) @This() {
            return .{
                .value = self.value / other.value,
                .derivative = (self.derivative * other.value - self.value * other.derivative) / (other.value * other.value),
            };
        }

        pub fn format(self: @This(), comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
            return writer.print("(value={}; df={})", .{ self.value, self.derivative });
        }
    };
}

/// Unit step function.
pub fn unit_step(in: anytype) @TypeOf(in) {
    switch (@typeInfo(@TypeOf(in))) {
        .Float, .Int, .ComptimeInt, .ComptimeFloat => {
            return @max(std.math.sign(in), 0);
        },
        .Struct => {
            if (@TypeOf(in) == Df32 or @TypeOf(in) == Df64) {
                return .{
                    .value = @max(std.math.sign(in.value), 0),
                    .derivative = @max(std.math.sign(in.value), 0) * in.derivative,
                };
            } else @compileError("Unsupported type");
        },
        else => @compileError("Unsupported type"),
    }
}

/// Exponential function.
pub fn exp(in: anytype) @TypeOf(in) {
    switch (@typeInfo(@TypeOf(in))) {
        .Float, .Int, .ComptimeInt, .ComptimeFloat => {
            return std.math.exp(in);
        },
        .Struct => {
            if (@TypeOf(in) == Df32 or @TypeOf(in) == Df64) {
                return .{
                    .value = std.math.exp(in.value),
                    .derivative = std.math.exp(in.value) * in.derivative,
                };
            } else @compileError("Unsupported type");
        },
        else => @compileError("Unsupported type"),
    }
}

/// Linear activation function
pub fn linear(in: anytype) @TypeOf(in) {
    switch (@typeInfo(@TypeOf(in))) {
        .Float, .Int, .ComptimeInt, .ComptimeFloat => {
            return in;
        },
        .Struct => {
            if (@TypeOf(in) == Df32 or @TypeOf(in) == Df64) {
                return in;
            } else @compileError("Unsupported type");
        },
        else => @compileError("Unsupported type"),
    }
}

/// ReLu (Rectified Linear Unit) activation function
pub fn relu(in: anytype) @TypeOf(in) {
    switch (@typeInfo(@TypeOf(in))) {
        .Float, .Int, .ComptimeInt, .ComptimeFloat => {
            return @max(0, in);
        },
        .Struct => {
            if (@TypeOf(in) == Df32 or @TypeOf(in) == Df64) {
                var val: @TypeOf(in).NUM_TYPE = 0.0;
                if (in.value > 0)
                    val = in.derivative;

                return .{
                    .value = @max(in.value, 0.0),
                    .derivative = val,
                };
            } else @compileError("Unsupported type");
        },
        else => @compileError("Unsupported type"),
    }
}

/// Sigmoid activation function
pub fn sigmoid(in: anytype) @TypeOf(in) {
    switch (@typeInfo(@TypeOf(in))) {
        .Float, .Int, .ComptimeInt, .ComptimeFloat => {
            return 1.0 / (1.0 + std.math.exp(-in));
        },
        .Struct => {
            if (@TypeOf(in) == Df32 or @TypeOf(in) == Df64) {
                const exp_val = 1.0 / (1.0 + std.math.exp(-in.value));
                return .{
                    .value = exp_val,
                    .derivative = exp_val * (1.0 - exp_val) * in.derivative,
                };
            } else @compileError("Unsupported type");
        },
        else => @compileError("Unsupported type"),
    }
}

/// Calculates the Mean Squared Error.
pub fn mse(a: []f32, b: []f32) f32 {
    std.debug.assert(a.len == b.len);

    var err: f32 = 0.0;
    for (a, b) |i, j|
        err += std.math.pow(f32, i - j, 2);

    return (1.0 / @as(f32, @floatFromInt(a.len))) * err;
}

test "auto diff derivative calc test" {
    const testing = std.testing;

    const test_func = &(struct {
        fn test_func(x: *const Df32) Df32 {
            return (Df32.literal(2.0).mul(x)).add(&Df32.literal(1.0));
        }
    }.test_func);

    const x = Df32.with_grad(5.0); // let x = 5.0 and compute the associated derivative

    const eval = test_func(&x);

    try testing.expectEqual(11.0, eval.value); //2 * (1) + 1 = 3
    try testing.expectEqual(2.0, eval.derivative);
}
