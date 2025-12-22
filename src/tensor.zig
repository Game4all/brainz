const std = @import("std");

/// Describes the shape of a tensor or an operation.
pub const Shape = struct {
    const Self = @This();
    /// Represents the layout strides for tensor with the current shape.
    const Strides = [MAX_DIMENSIONS]usize;

    /// Maximum number of dimensions allowed on a tensor
    const MAX_DIMENSIONS = 4;

    dimensions: [MAX_DIMENSIONS]usize,
    n_dimensions: usize,

    /// Returns a shape from a slice
    pub fn fromSlice(dims: []const usize) @This() {
        if (dims.len > MAX_DIMENSIONS) {
            if (@inComptime()) {
                @compileError(std.fmt.comptimePrint("Expected a shape slice with {} dimensions at most, got {} dimensions instead.", .{ MAX_DIMENSIONS, dims.len }));
            } else {
                @panic(std.fmt.comptimePrint("Expected a shape slice with {} dimensions at most, got more than {} dimensions.", .{ MAX_DIMENSIONS, MAX_DIMENSIONS }));
            }
        }

        var shape: Self = std.mem.zeroes(Self);
        shape.n_dimensions = dims.len;
        @memcpy(shape.dimensions[0..@min(dims.len, MAX_DIMENSIONS - 1)], dims);
        return shape;
    }

    /// Returns true if two shapes are equal
    pub fn eql(self: Self, other: Self) bool {
        if (self.n_dimensions != other.n_dimensions) return false;
        return std.mem.eql(usize, self.dimensions[0..self.n_dimensions], other.dimensions[0..other.n_dimensions]);
    }

    /// Returns the total count of elements in the shape.
    pub fn totalLength(self: *const Self) usize {
        if (self.n_dimensions == 0)
            return 0;

        var length: usize = 1;
        for (0..self.n_dimensions) |i| {
            length *= @max(self.dimensions[i], 1);
        }
        return length;
    }

    /// Returns the default layout strides for a shape
    pub fn layoutStrides(self: *const Self) Strides {
        var strides: Strides = std.mem.zeroes(Strides);
        if (self.n_dimensions == 0)
            return strides;

        strides[self.n_dimensions - 1] = 1;
        for (0..self.n_dimensions - 1) |i| {
            const dim = self.n_dimensions - 2 - i;
            strides[dim] = strides[dim + 1] * @max(self.dimensions[dim + 1], 1);
        }
        return strides;
    }
};

/// Possible data types of a tensor
pub const Dtype = enum {
    float32,
    float64,
    usize64,
};

/// A view over a slice of memory for computation. Essentialy, a multi dimensional matrix.
pub const Tensor = struct {
    /// tensor shape
    shape: Shape,
    /// tensor layout strides
    strides: Shape.Strides,
    /// tensor data type
    dtype: Dtype,
    /// pointer to tensor data
    data_ptr: *anyopaque,
    /// pointer to gradient tensor if one is computed for this tensor
    grad: ?*Tensor,

    /// Returns a typed pointer to underlying data.
    /// **The function may panic if the requested type is different from the tensor underlying data type**
    pub fn getData(self: *@This(), comptime ty: type) []ty {
        const target_dtype = switch (@typeInfo(ty)) {
            .float => |info| if (info.bits == 32) Dtype.float32 else Dtype.float64,
            .int => |info| if (info.bits == 64) Dtype.usize64 else @compileError("Unsupported integer type"),
            else => @compileError("Unsupported type for tensor data"),
        };

        if (self.dtype != target_dtype) @panic("Tensor dtype mismatch");

        const totalLength = self.shape.totalLength();

        const slice: [*]ty = @ptrCast(@alignCast(self.data_ptr));
        return slice[0..totalLength];
    }
};

/// Manages tensor allocation and storage
pub const TensorArena = struct {
    /// inner arena allocator
    inner_arena: std.heap.ArenaAllocator,

    pub fn init(allocator: std.mem.Allocator) TensorArena {
        return TensorArena{
            .inner_arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *TensorArena) void {
        self.inner_arena.deinit();
    }

    /// Creates a new tensor of specified data type and shape and allocates its storage.
    pub fn create(self: *TensorArena, dtype: Dtype, shape: Shape) !*Tensor {
        const totalElements = shape.totalLength();

        const elemSize: usize = switch (dtype) {
            .float32 => @sizeOf(f32),
            .float64 => @sizeOf(f64),
            .usize64 => @sizeOf(u64),
        };

        const allocator = self.inner_arena.allocator();

        const buffer = try allocator.alloc(u8, totalElements * elemSize);

        const tensor = try allocator.create(Tensor);
        tensor.* = Tensor{
            .shape = shape,
            .strides = shape.layoutStrides(),
            .dtype = dtype,
            .data_ptr = @ptrCast(buffer.ptr),
            .grad = null,
        };

        return tensor;
    }

    /// Creates a tensor that aliases the storage of another tensor.
    pub fn createAliased(self: *TensorArena, tensor: *Tensor) !*Tensor {
        const allocator = self.inner_arena.allocator();
        const alias = try allocator.create(Tensor);
        alias.* = tensor.*;
        return alias;
    }
};

// Tests

test "Shape.fromComptimeSlice creates correct shape" {
    const shape: Shape = comptime .fromSlice(&.{ 2, 3, 4 });
    try std.testing.expectEqual(3, shape.n_dimensions);

    try std.testing.expectEqual(2, shape.dimensions[0]);
    try std.testing.expectEqual(3, shape.dimensions[1]);
    try std.testing.expectEqual(4, shape.dimensions[2]);
}

test "Shape.fromComptimeSlice with 1D slice" {
    const shape: Shape = comptime .fromSlice(&.{5});
    try std.testing.expectEqual(1, shape.n_dimensions);
    try std.testing.expectEqual(5, shape.dimensions[0]);
}

test "Shape.eql returns true for equal shapes" {
    const shape1: Shape = comptime .fromSlice(&.{ 2, 3, 4 });
    const shape2: Shape = comptime .fromSlice(&.{ 2, 3, 4 });
    try std.testing.expect(shape1.eql(shape2));
}

test "Shape.eql returns false for different dimensions" {
    const shape1: Shape = comptime .fromSlice(&.{ 2, 3, 4 });
    const shape2: Shape = comptime .fromSlice(&.{ 2, 3 });
    try std.testing.expect(!shape1.eql(shape2));
}

test "Shape.Stride computed layouts are fine" {
    const shape: Shape = comptime .fromSlice(&.{ 2, 3, 4 });
    const strides = shape.layoutStrides();

    try std.testing.expectEqual(12, strides[0]);
    try std.testing.expectEqual(4, strides[1]);
    try std.testing.expectEqual(1, strides[2]);
}

test "TensorArena + Tensor allocation works" {
    var arena = TensorArena.init(std.testing.allocator);
    defer arena.deinit();

    var tensor = try arena.create(.float32, comptime .fromSlice(&.{ 2, 3, 4 }));

    try std.testing.expectEqual(3, tensor.shape.n_dimensions);
    try std.testing.expectEqual(24, tensor.shape.totalLength());

    const data: []f32 = tensor.getData(f32);
    try std.testing.expectEqual(tensor.shape.totalLength(), data.len);
}
