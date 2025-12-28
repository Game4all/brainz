const std = @import("std");
const tensor = @import("tensor.zig");

const Dtype = tensor.Dtype;
const Shape = tensor.Shape;
const Tensor = tensor.Tensor;
const TensorArena = tensor.TensorArena;
const Allocator = std.mem.Allocator;

/// Represents a virtual table for operations.
const OpInfo = struct {
    forward: *const fn (inputs: []const *Tensor, output: *Tensor, extra_data: ?*anyopaque) anyerror!void,
    backward: *const fn (inputs: []const *Tensor, output: *Tensor, grad_output: *Tensor, extra_data: ?*anyopaque) anyerror!void,
    /// The user-visible name of the operation for debugging purposes
    name: [:0]u8,
};

/// Represents an operation in the linear program.
const OpNode = struct {
    /// Information + VTable about the operation
    op_info: *const OpInfo,
    /// Inputs of the operation
    inputs: []const *Tensor,
    /// Output of the operation
    output: *Tensor,
    /// Pointer to additional extra data for the op.
    extra_data: ?*anyopaque,
};

/// Represents a linear execution plan with operations to be executed
const Program = struct {
    /// arena for
    arena: *TensorArena,
    /// allocator for storing program metadata and operations
    allocator: std.mem.Allocator,
    /// list of operations
    ops: std.ArrayList(OpNode),
    /// inputs of the program
    prog_inputs: std.StringArrayHashMapUnmanaged(*Tensor),
    /// outputs of the program
    prog_outputs: std.StringArrayHashMapUnmanaged(*Tensor),
    // tracks whether the graph was allocated
    allocated: bool,

    /// Initializes an empty program
    pub fn init(arena: *TensorArena, alloc: Allocator) @This() {
        return .{
            .allocator = alloc,
            .arena = arena,
            .prog_inputs = .empty,
            .prog_outputs = .empty,
            .ops = .empty,
            .allocated = false,
        };
    }

    /// Registers a tensor as an input.
    pub fn registerInput(self: *@This(), input_name: []const u8, input: *const Tensor) !void {
        try self.prog_inputs.put(self.allocator, input_name, input);
    }

    /// Creates a tensor as an input and registers it with the program
    pub fn createInput(self: *@This(), input_name: []const u8, dtype: Dtype, shape: Shape) !*Tensor {
        const t = try self.arena.makeTensor(dtype, shape);
        try self.prog_inputs.put(self.allocator, input_name, t);
        return t;
    }

    /// Retrieves a tensor by name from inputs or outputs
    pub fn getInput(self: *const @This(), name: []const u8) ?*Tensor {
        if (self.prog_inputs.get(name)) |ten|
            return ten;

        return null;
    }

    /// Registers a tensor as an output.
    pub fn registerOutput(self: *@This(), output_name: []const u8, output: *const Tensor) !void {
        try self.prog_outputs.put(self.allocator, output_name, output);
    }

    /// Creates a tensor as an output and registers it with the program
    pub fn createOutput(self: *@This(), output_name: []const u8, shape: Shape, dtype: Dtype) !*Tensor {
        const t = try self.arena.makeTensor(dtype, shape);
        try self.prog_outputs.put(self.allocator, output_name, t);
        return t;
    }

    /// Retrieves all outputs of the program
    pub fn getOutput(self: *const @This(), name: []const u8) ?*Tensor {
        if (self.prog_outputs.get(name)) |ten|
            return ten;

        return null;
    }

    /// Performs a forward pass of the program.
    pub fn forward(self: *@This()) !void {
        if (!self.allocated) {
            try self.arena.allocateStorage();
            self.allocated = true;
        }

        for (self.ops.items) |op_node| {
            try op_node.op_info.forward(
                op_node.inputs,
                op_node.output,
                op_node.extra_data,
            );
        }
    }

    /// Frees the memory backing the program
    pub fn deinit(self: *@This()) void {
        self.ops.deinit(self.allocator);
        self.prog_outputs.deinit(self.allocator);
        self.prog_inputs.deinit(self.allocator);
    }
};

test "creating an empty program" {
    var memArena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer memArena.deinit();

    // create tensor arena
    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    // create executable program
    var program: Program = .init(&tensorArena, memArena.allocator());
    defer program.deinit();

    _ = try program.createInput("input", .float32, comptime .fromSlice(&.{ 2, 3, 4 }));

    const i = program.getInput("input");

    try std.testing.expect(i != null);
}
