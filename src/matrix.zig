const std = @import("std");

const Random = std.rand.Random;
const Allocator = std.mem.Allocator;

/// A matrix with M rows and N columns.
pub fn Matrix(dtype: type) type {
    return struct {
        /// The dimensions of this matrix (M rows x N columns) + the strides for a row and column.
        shape: struct { usize, usize },
        // The matrix strides.
        strides: struct { usize, usize },
        /// The underlying storage of this matrix.
        storage: Storage(dtype),

        /// Creates a matrix with all its values set to the default value.
        pub fn empty(dims: struct { usize, usize }, allocator: Allocator) !@This() {
            const storage = try Storage(dtype).create_owned(dims, allocator);

            return @This(){
                .shape = dims,
                .strides = .{ dims.@"1", 1 },
                .storage = storage,
            };
        }

        /// Creates a matrix with all its values set to the given value.
        pub fn with_value(dims: struct { usize, usize }, def: dtype, allocator: Allocator) !@This() {
            var storage = try Storage(dtype).create_owned(dims, allocator);
            @memset(storage.get_mut_slice(), def);

            return @This(){
                .shape = dims,
                .strides = .{ dims.@"1", 1 },
                .storage = storage,
            };
        }

        /// Creates a matrix with all its values initialized with a RNG.
        pub fn random(dims: struct { usize, usize }, rng: Random, allocator: Allocator) !@This() {
            var storage = try Storage(dtype).create_owned(dims, allocator);
            for (storage.get_mut_slice()) |*val|
                val.* = rng.floatNorm(dtype);

            return @This(){
                .shape = dims,
                .strides = .{ dims.@"1", 1 },
                .storage = storage,
            };
        }

        /// Gets the value in the matrix at the specified position.
        pub inline fn get(self: *const @This(), pos: struct { usize, usize }) dtype {
            std.debug.assert(pos.@"0" < self.shape.@"0");
            std.debug.assert(pos.@"1" < self.shape.@"1");

            return self.storage.get(pos.@"0" * self.strides.@"0" + pos.@"1" * self.strides.@"1");
        }

        /// Sets the value in the matrix at the specified position.
        pub inline fn set(self: *@This(), pos: struct { usize, usize }, value: dtype) void {
            std.debug.assert(pos.@"0" < self.shape.@"0");
            std.debug.assert(pos.@"1" < self.shape.@"1");
            self.storage.set(pos.@"0" * self.strides.@"0" + pos.@"1" * self.strides.@"1", value);
        }

        /// Fills the matrix with the specified value.
        pub inline fn fill(self: *@This(), val: dtype) void {
            @memset(self.get_mut_slice(), val);
        }

        /// Fills the matrix with the specified slice.
        pub inline fn set_data(self: *@This(), data: []const dtype) void {
            std.debug.assert(data.len == self.get_slice().len);
            @memcpy(self.get_mut_slice(), data);
        }

        /// Returns the transposed matrix.
        /// NOTE: The transposed matrix shares the same storage as the original matrix.
        pub inline fn transpose(self: *const @This()) @This() {
            return .{
                .shape = .{ self.shape.@"1", self.shape.@"0" },
                .strides = .{ self.strides.@"1", self.strides.@"0" },
                .storage = self.storage.as_view(),
            };
        }

        /// Returns a const slice representing the contents of this matrix.
        pub inline fn get_slice(self: *const @This()) []const dtype {
            return self.storage.get_slice();
        }

        /// Returns a mutable slice representing the contents of this matrix.
        pub inline fn get_mut_slice(self: *@This()) []dtype {
            return self.storage.get_mut_slice();
        }

        /// Frees the values.
        pub fn deinit(self: *@This()) void {
            self.storage.deinit();
        }

        // Format
        pub fn format(self: *const @This(), comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
            const shape = self.shape;
            var dat = @constCast(self);
            _ = try writer.print("\n\r[\n\r", .{});
            for (0..shape[0]) |i| {
                try writer.print(" [", .{});
                for (0..shape[1]) |j| {
                    _ = try writer.print(" {e:.03}", .{dat.get(.{ i, j })});
                    _ = try writer.write(",");
                }
                try writer.print("]\n\r", .{});
            }
            try writer.print("]", .{});
        }
    };
}

/// Storage for matrix data.
pub fn Storage(dtype: type) type {
    return struct {
        const Self = @This();

        /// Dimensions of the matrix storage (M rows x N columns).
        dimensions: struct { usize, usize },
        /// The storage kind (owns memory or is simply a view to another storage).
        data: union(enum) {
            /// The memory is owned.
            Owned: struct {
                allocator: Allocator,
                data: []dtype,
            },
            /// A view into another matrix storage.
            View: *Self,
        },

        /// Creates a matrix storage which owns the memory.
        pub inline fn create_owned(dims: struct { usize, usize }, allocator: Allocator) !@This() {
            const dimensions = blk: {
                if (dims.@"0" * dims.@"1" == 0)
                    return error.InvalidStorageSize;

                break :blk dims;
            };

            return .{
                .dimensions = dimensions,
                .data = .{
                    .Owned = .{
                        .allocator = allocator,
                        .data = try allocator.alloc(dtype, dims.@"0" * dims.@"1"),
                    },
                },
            };
        }

        /// Returns a view pointing to this storage.
        pub inline fn as_view(self: *const @This()) @This() {
            return .{
                .dimensions = self.dimensions,
                .data = switch (self.data) {
                    .Owned => .{
                        .View = @constCast(self),
                    },
                    .View => |view| .{
                        .View = view,
                    },
                },
            };
        }

        /// Attempts to get the value at specified index in the matrix storage.
        pub fn get(self: *const @This(), idx: usize) dtype {
            return switch (self.data) {
                .Owned => |own| own.data[idx],
                .View => |view| view.get(idx),
            };
        }

        /// Attempts to set the value at the specified index in the matrix storage.
        pub fn set(self: *@This(), idx: usize, data: dtype) void {
            return switch (self.data) {
                .Owned => |own| own.data[idx] = data,
                .View => |view| view.set(idx, data),
            };
        }

        /// Returns the storage's memory as a const slice.
        pub fn get_slice(self: *const @This()) []const dtype {
            return switch (self.data) {
                .Owned => |own| own.data,
                .View => |view| view.get_slice(),
            };
        }

        pub fn get_mut_slice(self: *@This()) []dtype {
            return switch (self.data) {
                .Owned => |own| own.data,
                .View => |view| view.get_mut_slice(),
            };
        }

        /// Deinitializes the storage if it owns memory.
        pub inline fn deinit(self: *@This()) void {
            switch (self.data) {
                .Owned => |own| own.allocator.free(own.data),
                else => {},
            }
        }
    };
}

test "matrix indexing + stride test" {
    // create a 3x3 square matrix.

    // default stride (3, 1)
    var mat = try Matrix(f32).with_value(.{ 3, 3 }, 0, std.testing.allocator);
    defer mat.deinit();

    for (0..3) |value|
        mat.set(.{ value, value }, @floatFromInt(value));

    mat.set(.{ 0, 2 }, 3.0);

    // matrix state at this point
    // 0 0 3
    // 0 1 0
    // 0 0 2

    try std.testing.expectEqual(0, mat.get(.{ 0, 0 }));
    try std.testing.expectEqual(1, mat.get(.{ 1, 1 }));
    try std.testing.expectEqual(2, mat.get(.{ 2, 2 }));
    try std.testing.expectEqual(0, mat.get(.{ 2, 0 }));

    // transpose the matrix
    // strides becomes (1, 3)
    // matrix view is
    // 0 0 0
    // 0 1 0
    // 3 0 2
    var transposed = mat.transpose();
    try std.testing.expectEqual(3, transposed.get(.{ 2, 0 }));
    try std.testing.expectEqual(0, transposed.get(.{ 0, 2 }));
}
