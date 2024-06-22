const std = @import("std");

const Random = std.rand.Random;
const Allocator = std.mem.Allocator;

/// A Tensor with NxMxL dimensions
///
/// NOTE: Tensors should be allocated with an arena allocator as they are designed to be all pre-allocated and freed at once.
pub fn Tensor(dtype: type) type {
    return struct {
        /// The dimensions of tensor matrix (N batches x M rows x N columns)
        shape: struct { usize, usize, usize },
        // The tensor strides.
        strides: struct { usize, usize, usize },
        /// The underlying storage of this tensor.
        storage: Storage(dtype),

        /// Creates a tensor with all its values set to the default value.
        pub fn empty(dims: struct { usize, usize, usize }, allocator: Allocator) !@This() {
            return @This(){
                .shape = dims,
                .strides = .{ @max(dims.@"0", 1) * @max(dims.@"1", 1), @max(dims.@"2", 1), 1 },
                .storage = try Storage(dtype).createOwned(dims, allocator),
            };
        }

        /// Creates a tensor with all its values set to the given value.
        pub fn withValue(dims: struct { usize, usize, usize }, def: dtype, allocator: Allocator) !@This() {
            var mat = try empty(dims, allocator);
            mat.fill(def);
            return mat;
        }

        /// Creates a tensor with all its values initialized with a RNG.
        pub fn random(dims: struct { usize, usize, usize }, rng: Random, allocator: Allocator) !@This() {
            var mat = try empty(dims, allocator);
            for (mat.slice()) |*val|
                val.* = rng.floatNorm(dtype);
            return mat;
        }

        /// Gets the value in the tensor at the specified position.
        pub inline fn get(self: *const @This(), pos: struct { usize, usize, usize }) dtype {
            std.debug.assert(pos.@"0" < @max(self.shape.@"0", 1));
            std.debug.assert(pos.@"1" < @max(self.shape.@"1", 1));
            std.debug.assert(pos.@"2" < @max(self.shape.@"2", 1));

            return self.storage.get(pos.@"0" * self.strides.@"0" + pos.@"1" * self.strides.@"1" + pos.@"2" * self.strides.@"2");
        }

        /// Sets the value in the tensor at the specified position.
        pub inline fn set(self: *@This(), pos: struct { usize, usize, usize }, value: dtype) void {
            std.debug.assert(pos.@"0" < @max(self.shape.@"0", 1));
            std.debug.assert(pos.@"1" < @max(self.shape.@"1", 1));
            std.debug.assert(pos.@"2" < @max(self.shape.@"2", 1));

            self.storage.set(pos.@"0" * self.strides.@"0" + pos.@"1" * self.strides.@"1" + pos.@"2" * self.strides.@"2", value);
        }

        /// Fills the tensor with the specified value.
        pub inline fn fill(self: *@This(), val: dtype) void {
            @memset(self.slice(), val);
        }

        /// Fills the tensor with the specified slice.
        pub inline fn setData(self: *@This(), data: []const dtype) void {
            std.debug.assert(data.len == self.constSlice().len);
            @memcpy(self.slice(), data);
        }

        /// Returns the transposed tensor.
        /// NOTE: The transposed tensor shares the same storage as the original tensor.
        pub inline fn transpose(self: *const @This()) @This() {
            return .{
                .shape = .{ self.shape.@"0", self.shape.@"2", self.shape.@"1" },
                .strides = .{ self.strides.@"0", self.strides.@"2", self.strides.@"1" },
                .storage = self.storage.asView(),
            };
        }

        /// Reshapes this tensor.
        /// Returns a view to the reshaped tensor.
        /// NOTE: This doesn't resize the underlying tensor storage so the new size length should be smaller as big as the original shape.
        pub inline fn reshape(self: *const @This(), new_dims: struct { usize, usize, usize }) @This() {
            std.debug.assert(@max(1, new_dims.@"2") * @max(1, new_dims.@"1") * @max(1, new_dims.@"0") == @max(1, self.shape.@"2") * @max(1, self.shape.@"1") * @max(1, self.shape.@"0"));
            return .{
                .shape = new_dims,
                .strides = .{ @max(new_dims.@"0", 1) * @max(new_dims.@"1", 1), @max(new_dims.@"2", 1), 1 },
                .storage = self.storage.asView(),
            };
        }

        /// Returns a const slice representing the contents of this tensor.
        pub inline fn constSlice(self: *const @This()) []const dtype {
            return self.storage.constSlice();
        }

        /// Returns a mutable slice representing the contents of this tensor.
        pub inline fn slice(self: *@This()) []dtype {
            return self.storage.slice();
        }

        /// Frees the values.
        pub fn deinit(self: *@This(), allocator: Allocator) void {
            self.storage.deinit(allocator);
        }

        // Format
        pub fn format(self: *const @This(), comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
            const shape = self.shape;
            var dat = @constCast(self);

            try writer.print("[\n\r", .{});
            for (0..@max(shape[0], 1)) |h| {
                _ = try writer.print(" [\n\r", .{});
                for (0..@max(shape[1], 1)) |i| {
                    try writer.print("   [", .{});
                    for (0..@max(shape[2], 1)) |j| {
                        _ = try writer.print(" {e:.03}", .{dat.get(.{ h, i, j })});
                        _ = try writer.write(",");
                    }
                    try writer.print(" ],\n\r", .{});
                }
                try writer.print(" ],\n\r", .{});
            }
            try writer.print("]", .{});
        }
    };
}

/// Storage for tensor data.
fn Storage(dtype: type) type {
    return union(enum) {
        const Self = @This();

        /// The memory is owned.
        Owned: []dtype,
        /// A view into another tensor storage.
        View: *Self,

        /// Creates a tensor storage which owns the memory.
        inline fn createOwned(dims: struct { usize, usize, usize }, allocator: Allocator) !@This() {
            if (dims.@"1" == 0 and dims.@"2" == 0 and dims.@"0" == 0)
                return error.InvalidStorageSize;

            return .{
                .Owned = try allocator.alloc(dtype, @max(1, dims.@"2") * @max(1, dims.@"1") * @max(1, dims.@"0")),
            };
        }

        /// Returns a view pointing to this storage.
        inline fn asView(self: *const @This()) @This() {
            return switch (self.*) {
                .Owned => .{
                    .View = @constCast(self),
                },
                .View => |view| .{
                    .View = view,
                },
            };
        }

        /// Attempts to get the value at specified index in the tensor storage.
        fn get(self: *const @This(), idx: usize) dtype {
            return switch (self.*) {
                .Owned => |own| own[idx],
                .View => |view| view.get(idx),
            };
        }

        /// Attempts to set the value at the specified index in the tensor storage.
        fn set(self: *@This(), idx: usize, data: dtype) void {
            return switch (self.*) {
                .Owned => |own| own[idx] = data,
                .View => |view| view.set(idx, data),
            };
        }

        /// Returns the storage's memory as a const slice.
        fn constSlice(self: *const @This()) []const dtype {
            return switch (self.*) {
                .Owned => |own| own,
                .View => |view| view.constSlice(),
            };
        }

        fn slice(self: *@This()) []dtype {
            return switch (self.*) {
                .Owned => |own| own,
                .View => |view| view.slice(),
            };
        }

        /// Deinitializes the storage if it owns memory.
        /// Use this if you haven't allocated this tensor using an arena allocator and need exact control over when the deallocation happens.
        inline fn deinit(self: *@This(), allocator: Allocator) void {
            switch (self.*) {
                .Owned => |own| allocator.free(own),
                else => {},
            }
        }
    };
}

test "tensor indexing + stride test" {
    // create a 3x3 square tensor.

    // default stride (3, 1)
    var mat = try Tensor(f32).withValue(.{ 0, 3, 3 }, 0, std.testing.allocator);
    defer mat.deinit(std.testing.allocator);

    for (0..3) |value|
        mat.set(.{ 0, value, value }, @floatFromInt(value));

    mat.set(.{ 0, 0, 2 }, 3.0);

    // tensor state at this point
    // 0 0 3
    // 0 1 0
    // 0 0 2

    try std.testing.expectEqual(0, mat.get(.{ 0, 0, 0 }));
    try std.testing.expectEqual(1, mat.get(.{ 0, 1, 1 }));
    try std.testing.expectEqual(2, mat.get(.{ 0, 2, 2 }));
    try std.testing.expectEqual(0, mat.get(.{ 0, 2, 0 }));

    // transpose the tensor
    // strides becomes (1, 3)
    // tensor view is
    // 0 0 0
    // 0 1 0
    // 3 0 2
    var transposed = mat.transpose();
    try std.testing.expectEqual(3, transposed.get(.{ 0, 2, 0 }));
    try std.testing.expectEqual(0, transposed.get(.{ 0, 0, 2 }));
}

test "tensor alloc test" {
    var mat1 = try Tensor(f32).withValue(.{ 3, 0, 3 }, 0, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    var mat2 = try Tensor(f32).withValue(.{ 3, 3, 3 }, 0, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);

    try std.testing.expectEqual(9, mat1.constSlice().len);
    try std.testing.expectEqual(27, mat2.constSlice().len);
}
