const std = @import("std");
const Allocator = std.mem.Allocator;

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

    /// Returns a new shape expanded with the given batch dimensions prepended
    pub fn expandDims(self: Self, batch_dims: []const usize) Self {
        const new_n_dims = self.n_dimensions + batch_dims.len;
        if (new_n_dims > MAX_DIMENSIONS) {
            if (@inComptime()) {
                @compileError(std.fmt.comptimePrint("Expected the final shape to have {} dimensions at most, got {} dimensions instead.", .{ MAX_DIMENSIONS, new_n_dims }));
            } else {
                @panic(std.fmt.comptimePrint("Expected the final shape to have {} dimensions at most, got more than {} dimensions.", .{ MAX_DIMENSIONS, MAX_DIMENSIONS }));
            }
        }

        var new_shape: Self = std.mem.zeroes(Self);
        new_shape.n_dimensions = new_n_dims;

        @memcpy(new_shape.dimensions[0..batch_dims.len], batch_dims);
        @memcpy(new_shape.dimensions[batch_dims.len..new_n_dims], self.dimensions[0..self.n_dimensions]);

        return new_shape;
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

    /// Returns the "batch" dimension of the shape which is assumed to be the first dimension, representing the number of samples processed at once per operation.
    pub inline fn batchDim(self: *const Self) usize {
        return if (self.n_dimensions > 0) self.dimensions[0] else 1;
    }
};

/// Possible data types of a tensor
pub const Dtype = enum {
    float32,
    float64,
    usize64,

    /// Returns whether the current Dtype is a floating point type
    pub fn isFloatingPoint(self: Dtype) bool {
        return switch (self) {
            .float32, .float64 => true,
            else => false,
        };
    }
};

const dtype_to_size = std.enums.EnumArray(Dtype, usize).init(.{
    .float32 = @sizeOf(f32),
    .float64 = @sizeOf(f64),
    .usize64 = @sizeOf(u64),
});

/// Represents a logical tensor (essentially a multi-dimensional matrix) which may be backed by physical memory.
pub const Tensor = struct {
    /// tensor shape
    shape: Shape,
    /// tensor layout strides
    strides: Shape.Strides,
    /// tensor data type
    dtype: Dtype,
    /// pointer to the storage backing that tensor
    storage: ?*anyopaque,
    /// whether this tensor requires computing gradients.
    requires_grad: bool,
    /// pointer to gradient tensor if one is computed for this tensor
    grad: ?*const Tensor,
    /// Pointer to the parent tensor, if a view.
    parent_view: ?*const Tensor,

    /// Returns whether a Tensor is a view of another tensor.
    pub inline fn isView(self: *const Tensor) bool {
        return self.parent_view != null;
    }

    /// Returns whether this tensor is backed by memory to be used in computation.
    pub inline fn hasStorage(self: *const Tensor) bool {
        return self.storage != null;
    }

    fn getRawStorage(self: *const Tensor) ?*anyopaque {
        if (self.storage) |s| return s;
        if (self.parent_view) |p| return p.getRawStorage();
        return null;
    }

    //todo: enforce type safety
    /// Returns the tensor storage as a slice of the specified type.
    pub fn slice(self: *const Tensor, comptime T: type) ?[]T {
        const storage = self.getRawStorage() orelse return null;
        const ptr: [*]T = @ptrCast(@alignCast(storage));
        const sl = ptr[0..self.shape.totalLength()];
        return sl;
    }

    /// Returns the tensor storage as a scalar of the specified type.
    pub fn scalar(self: *const Tensor, comptime T: type) ?T {
        if (self.shape.totalLength() != 1) return null;
        const sl = self.slice(T) orelse return null;
        return sl[0];
    }

    //todo: this may have to live on the Device interface at some point
    /// Resets the tensor storage to zero if it exists.
    pub fn zero(self: *const Tensor) void {
        switch (self.dtype) {
            .float32 => if (self.slice(f32)) |s| @memset(s, 0),
            .float64 => if (self.slice(f64)) |s| @memset(s, 0),
            .usize64 => if (self.slice(u64)) |s| @memset(s, 0),
        }
    }
};

/// Tracks tensor lifetimes and manages view aliasing for a single graph, enabling efficient allocation and reuse of tensors' backing memory.
pub const TensorArena = struct {
    allocator: Allocator,
    tensors: std.ArrayList(*Tensor),

    pub inline fn init(allocator: Allocator) TensorArena {
        return TensorArena{ .allocator = allocator, .tensors = .empty };
    }

    /// Frees all allocations made (including tensors + allocated storages)
    pub fn deinit(self: *TensorArena) void {
        for (self.tensors.items) |tensor| {
            if (tensor.isView()) continue;

            if (tensor.storage) |storage| {
                //FIXME: fix storage freeing
                const byte_size: usize = tensor.shape.totalLength() * dtype_to_size.get(tensor.dtype);
                const typed_storage = @as([*]u8, @ptrCast(storage));
                self.allocator.free(typed_storage[0..byte_size]);
            }
        }
        self.tensors.deinit(self.allocator);
    }

    /// Creates a new tensor of specified data type and shape.
    pub fn makeTensor(self: *TensorArena, dtype: Dtype, shape: Shape, requires_grad: bool) !*const Tensor {
        const tensor = try self.allocator.create(Tensor);
        tensor.* = Tensor{
            .shape = shape,
            .strides = shape.layoutStrides(),
            .dtype = dtype,
            .storage = null,
            .requires_grad = requires_grad,
            .grad = null,
            .parent_view = null,
        };

        try self.tensors.append(self.allocator, tensor);

        return tensor;
    }

    /// Creates a new tensor as a view of a parent tensor.
    pub fn makeView(self: *TensorArena, parent: *const Tensor, shape: Shape) !*const Tensor {
        const tensor = try self.makeTensor(parent.dtype, shape, false);
        const T: *Tensor = @constCast(tensor);
        T.parent_view = parent;
        return tensor;
    }

    /// Allocates backing storage for all non-view tensors in the arena.
    /// # Note
    /// You should ONLY call that after finalizing your compute plan(s) to allocate memory for the tensors storage.
    pub fn allocateStorage(self: *TensorArena) !void {
        for (self.tensors.items) |tensor| {
            if (tensor.isView()) continue;

            const byte_size: usize = tensor.shape.totalLength() * dtype_to_size.get(tensor.dtype);
            const storage = try self.allocator.alloc(u8, byte_size);
            tensor.storage = storage.ptr;
        }
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

test "Shape.expandDims prepends dimensions" {
    const shape: Shape = comptime .fromSlice(&.{ 3, 4 });
    const batch_shape = shape.expandDims(&.{ 2, 5 });

    try std.testing.expectEqual(4, batch_shape.n_dimensions);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 5, 3, 4 }, batch_shape.dimensions[0..batch_shape.n_dimensions]);
}

test "Shape.expandDims at comptime" {
    const shape: Shape = comptime .fromSlice(&.{4});
    const batch_shape: Shape = comptime shape.expandDims(&.{ 2, 3 });

    try std.testing.expectEqual(3, batch_shape.n_dimensions);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 4 }, batch_shape.dimensions[0..batch_shape.n_dimensions]);
}

test "TensorArena test" {
    var memArena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    const tensor1 = try tensorArena.makeTensor(.float32, comptime .fromSlice(&.{ 2, 3 }), false);
    const view1 = try tensorArena.makeView(tensor1, comptime .fromSlice(&.{ 3, 2 }));

    try std.testing.expectEqual(null, tensor1.slice(f32)); // storage should be null until this point since no backing memory has been allocated

    try tensorArena.allocateStorage(); // bam, we allocate storage for all tensors

    const slice = tensor1.slice(f32);

    try std.testing.expect(slice != null); // storage should now exist
    @memset(slice.?, 123);

    const viewSlice = view1.slice(f32);
    try std.testing.expect(viewSlice != null);

    try std.testing.expectEqual(123, viewSlice.?[0]);

    try std.testing.expect(tensorArena.tensors.items[0].storage != null);
    try std.testing.expect(tensorArena.tensors.items[1].storage == null);
}
