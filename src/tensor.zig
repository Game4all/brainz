const std = @import("std");

const Random = std.rand.Random;
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

/// A Tensor with NxMxL dimensions.
/// A tensor is simply a view into a block of memory such that "views" of a tensor are tensors pointing to the **same memory block** (with different strides).
/// NOTE: Tensors don't automatically manage ownership of the memory they point to and as such the memory management is up to the programmer (use an ArenaAllocator).
pub fn Tensor(dtype: type) type {
    return struct {
        /// The dimensions of the tensor (N batches x M rows x N columns)
        shape: struct { usize, usize, usize },
        // The tensor strides.
        strides: struct { usize, usize, usize },
        /// Pointer to the data of this tensor.
        data: []dtype,

        /// Creates a tensor of specified dimensions from the specified slice.
        /// NOTE: The created tensor **doesn't own** the slice and care must be taken by the programmer to deallocate it.
        pub fn initFromSlice(dims: struct { usize, usize, usize }, data: []dtype) !@This() {
            if (std.meta.eql(dims, .{ 0, 0, 0 }))
                return error.InvalidStorageSize;

            if (@max(1, dims.@"2") * @max(1, dims.@"1") * @max(1, dims.@"0") != data.len)
                return error.InvalidSliceSize;

            return .{
                .shape = dims,
                .strides = .{ @max(dims.@"2", 1) * @max(dims.@"1", 1), @max(dims.@"2", 1), 1 },
                .data = data,
            };
        }

        /// Creates a tensor and allocate its storage using the specified allocator.
        /// See `deinit` to deallocate the tensor storage.
        pub fn init(dims: struct { usize, usize, usize }, allocator: Allocator) !@This() {
            const memory = try allocator.alloc(dtype, @max(1, dims.@"2") * @max(1, dims.@"1") * @max(1, dims.@"0"));
            errdefer allocator.free(memory);

            return try initFromSlice(dims, memory);
        }

        /// Gets the value in the tensor at the specified position.
        pub inline fn get(self: *const @This(), pos: struct { usize, usize, usize }) dtype {
            std.debug.assert(pos.@"0" < @max(self.shape.@"0", 1));
            std.debug.assert(pos.@"1" < @max(self.shape.@"1", 1));
            std.debug.assert(pos.@"2" < @max(self.shape.@"2", 1));

            return self.data[pos.@"0" * self.strides.@"0" + pos.@"1" * self.strides.@"1" + pos.@"2" * self.strides.@"2"];
        }

        /// Sets the value in the tensor at the specified position.
        pub inline fn set(self: *@This(), pos: struct { usize, usize, usize }, value: dtype) void {
            std.debug.assert(pos.@"0" < @max(self.shape.@"0", 1));
            std.debug.assert(pos.@"1" < @max(self.shape.@"1", 1));
            std.debug.assert(pos.@"2" < @max(self.shape.@"2", 1));

            self.data[pos.@"0" * self.strides.@"0" + pos.@"1" * self.strides.@"1" + pos.@"2" * self.strides.@"2"] = value;
        }

        /// Fills the tensor with the specified value.
        pub inline fn fill(self: *@This(), val: dtype) void {
            @memset(self.slice(), val);
        }

        /// Fills the tensor with random values
        pub fn fillRandom(self: *@This(), rng: Random) void {
            for (self.slice()) |*val|
                val.* = rng.floatNorm(dtype);
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
                .data = self.data,
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
                .data = self.data,
            };
        }

        /// Returns a const slice representing the contents of this tensor.
        pub inline fn constSlice(self: *const @This()) []const dtype {
            return self.data;
        }

        /// Returns a mutable slice representing the contents of this tensor.
        pub inline fn slice(self: *@This()) []dtype {
            return self.data;
        }

        /// Checks that the elements of this tensor are contiguous in memory.
        pub inline fn isContiguous(self: *const @This()) bool {
            return self.strides.@"0" >= self.strides.@"1" and self.strides.@"1" >= self.strides.@"2";
        }

        /// Frees the memory this tensor holds.
        pub fn deinit(self: *@This(), allocator: Allocator) void {
            allocator.free(self.data);
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
                        switch (@typeInfo(dtype)) {
                            .Float => |_| try writer.print(" {e:.03}", .{dat.get(.{ h, i, j })}),
                            else => |_| try writer.print(" {}", .{dat.get(.{ h, i, j })}),
                        }
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

test "tensor indexing + stride test" {
    // create a 3x3 square tensor.

    // default stride (3, 1)
    var mat = try Tensor(f32).init(.{ 0, 3, 3 }, std.testing.allocator);
    mat.fill(0.0);

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
    var mat1 = try Tensor(f32).init(.{ 3, 0, 3 }, std.testing.allocator);
    defer mat1.deinit(std.testing.allocator);

    var mat2 = try Tensor(f32).init(.{ 3, 3, 3 }, std.testing.allocator);
    defer mat2.deinit(std.testing.allocator);

    try std.testing.expectEqual(9, mat1.constSlice().len);
    try std.testing.expectEqual(27, mat2.constSlice().len);
}
