const std = @import("std");
const tensor = @import("tensor.zig");

const Dtype = tensor.Dtype;
const Shape = tensor.Shape;
const Tensor = tensor.Tensor;
const TensorArena = tensor.TensorArena;
const Allocator = std.mem.Allocator;

/// Represents a virtual table for operations.
pub const OpInfo = struct {
    /// Performs the forward pass for the specified operation, writing the result into the `output` tensor.
    forward: *const fn (inputs: []const *const Tensor, output: *const Tensor, extra_data: ?*anyopaque) anyerror!void,
    /// Performs the backward pass for the specified operation, writing the gradients into the input tensors from the `grad_output` tensor (which is the gradient of the output tensor `output.grad`)
    backward: *const fn (inputs: []const *const Tensor, output: *const Tensor, grad_output: *const Tensor, extra_data: ?*anyopaque) anyerror!void,
    /// The user-visible name of the operation for debugging purposes
    name: [:0]const u8,
};

/// Represents an operation in the linear program.
const OpNode = struct {
    /// Max number of allowed inputs for a single operation
    const MAX_INPUTS = 2;

    /// Information + VTable about the operation
    op_info: *const OpInfo,
    /// Inputs of the operation
    inputs: [MAX_INPUTS]*const Tensor,
    /// Numbers of input tensors
    n_inputs: usize = 0,
    /// Output of the operation
    output: *const Tensor,
    /// Pointer to additional extra data for the op.
    extra_data: ?*anyopaque,
};

/// Represents a linear execution plan with operations to be executed
pub const Program = struct {
    const Flags = packed struct(u2) {
        finalized: bool = false,
        allow_backprop: bool = false,
    };

    /// The arena managing the tensors of this graph.
    /// This arena may be used to create intermediate tensors for the operation results.
    arena: *TensorArena,
    /// allocator for storing program metadata and operations
    allocator: std.mem.Allocator,
    /// list of operations
    ops: std.ArrayList(OpNode),
    /// inputs of the program
    prog_inputs: std.StringArrayHashMapUnmanaged(*const Tensor),
    /// outputs of the program
    prog_outputs: std.StringArrayHashMapUnmanaged(*const Tensor),
    /// parameters of the program
    prog_params: std.ArrayList(*const Tensor),
    // internal flags for tracking program compilation state
    flags: Flags,

    /// Initializes an empty program
    pub fn init(arena: *TensorArena, alloc: Allocator) @This() {
        return .{
            .allocator = alloc,
            .arena = arena,
            .prog_inputs = .empty,
            .prog_outputs = .empty,
            .prog_params = .empty,
            .ops = .empty,
            .flags = .{},
        };
    }

    /// Appends an operation to the program.
    /// # Note
    /// This is a low-level operation, and you should use the operations in the `ops` module instead.
    pub fn append(self: *@This(), op_info: *const OpInfo, inputs: []const *const Tensor, out: *const Tensor, extra: ?*anyopaque) !void {
        if (self.flags.finalized) return error.ProgramIsFinalized;
        if (inputs.len > OpNode.MAX_INPUTS) {
            if (@inComptime()) {
                @compileError(std.fmt.comptimePrint("Expected at most {} inputs, got {} inputs instead.", .{ OpNode.MAX_INPUTS, inputs.len }));
            } else {
                return error.TooManyInputs;
            }
        }

        var node: OpNode = undefined;
        node.n_inputs = inputs.len;
        node.output = out;
        node.op_info = op_info;
        node.extra_data = extra;
        @memcpy(node.inputs[0..@min(OpNode.MAX_INPUTS, inputs.len)], inputs);

        try self.ops.append(self.allocator, node);
    }

    // ================================================= Tensor APIs ==================================================

    /// Registers a tensor as an input.
    pub fn registerInput(self: *@This(), input_name: []const u8, input: *const Tensor) !void {
        if (self.flags.finalized) return error.ProgramIsFinalized;
        try self.prog_inputs.put(self.allocator, input_name, input);
    }

    /// Creates a tensor as an input and registers it with the program
    pub fn createInput(self: *@This(), input_name: []const u8, dtype: Dtype, shape: Shape, require_grad: bool) !*const Tensor {
        if (self.flags.finalized) return error.ProgramIsFinalized;
        const t = try self.arena.makeTensor(dtype, shape, require_grad);
        try self.prog_inputs.put(self.allocator, input_name, t);
        return t;
    }

    /// Retrieves a tensor by name from inputs.
    pub fn getInput(self: *const @This(), name: []const u8) ?*const Tensor {
        if (self.prog_inputs.get(name)) |ten|
            return ten;

        return null;
    }

    /// Registers a tensor as an output.
    pub fn registerOutput(self: *@This(), output_name: []const u8, output: *const Tensor) !void {
        if (self.flags.finalized) return error.ProgramIsFinalized;
        try self.prog_outputs.put(self.allocator, output_name, output);
    }

    /// Creates a tensor as an output and registers it with the program
    pub fn createOutput(self: *@This(), output_name: []const u8, dtype: Dtype, shape: Shape, require_grad: bool) !*const Tensor {
        if (self.flags.finalized) return error.ProgramIsFinalized;
        const t = try self.arena.makeTensor(dtype, shape, require_grad);
        try self.prog_outputs.put(self.allocator, output_name, t);
        return t;
    }

    /// Retrieves all outputs of the program
    pub fn getOutput(self: *const @This(), name: []const u8) ?*const Tensor {
        if (self.prog_outputs.get(name)) |ten|
            return ten;

        return null;
    }

    /// Creates a tensor and registers it as an optimizable parameter for the program.
    pub fn createParam(self: *@This(), dtype: Dtype, shape: Shape) !*const Tensor {
        if (self.flags.finalized) return error.ProgramIsFinalized;
        const t = try self.arena.makeTensor(dtype, shape, true); // we consider parameters are optimizable by default
        try self.prog_params.append(self.allocator, t);
        return t;
    }

    /// Returns all the parameters of this program.
    pub inline fn getParams(self: *const @This()) []*const Tensor {
        return self.prog_params.items;
    }

    // ======================================================= Program finalization and passes ============================================

    /// Finalizes a program for execution.
    /// # Args
    /// - `backprop`: indicates whether to allocate gradient tensors for backpropagation.
    /// All tensors which have `require_grad` set to true will have a grad tensor attached to them for back propagation if `backprop` is enabled.
    pub fn finalize(self: *@This(), backprop: bool) !void {
        if (self.flags.finalized)
            return error.AlreadyFinalized;

        // program is now considered finalized and can't be mutated anymore.
        self.flags.finalized = true;
        // allow user to call backprop
        self.flags.allow_backprop = backprop;

        // allocate grad tensors for tensors which require a gradient, aren't views and do not have a gradient tensor already iff backprop is enabled
        if (backprop) {
            for (self.ops.items) |op| {
                const out = @constCast(op.output);

                if (out.requires_grad and !out.isView() and out.grad == null)
                    out.grad = try self.arena.makeTensor(out.dtype, out.shape, false);

                for (op.inputs) |in| {
                    if (in.requires_grad and !in.isView() and in.grad == null) {
                        const inT: *Tensor = @constCast(in);
                        inT.grad = try self.arena.makeTensor(in.dtype, in.shape, false);
                    }
                }
            }
        }
    }

    /// Performs a forward pass of the program.
    pub fn forward(self: *@This()) !void {
        if (!self.flags.finalized)
            return error.NotFinalized;

        for (self.ops.items) |op_node| {
            try op_node.op_info.forward(
                op_node.inputs[0..op_node.n_inputs],
                op_node.output,
                op_node.extra_data,
            );
        }
    }

    /// Resets gradients of all tensors attached to the program to zero.
    /// # Note
    /// - You should call this before calling `backward()` to accumulate gradients properly.
    pub fn zeroGrad(self: *@This()) void {
        for (self.ops.items) |op| {
            if (op.output.grad) |grad|
                grad.zero();

            for (op.inputs[0..op.n_inputs]) |input| {
                if (input.grad) |grad|
                    grad.zero();
            }
        }
    }

    /// Performs a backward pass of the program.
    pub fn backward(self: *@This()) !void {
        if (!self.flags.finalized) return error.NotFinalized;
        if (!self.flags.allow_backprop) return error.BackpropNotEnabled;

        var i: usize = self.ops.items.len;
        while (i > 0) {
            i -= 1;
            const op_node = self.ops.items[i];
            if (op_node.output.grad) |grad_output| {
                try op_node.op_info.backward(
                    op_node.inputs[0..op_node.n_inputs],
                    op_node.output,
                    grad_output,
                    op_node.extra_data,
                );
            }
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

    _ = try program.createInput("input", .float32, comptime .fromSlice(&.{ 2, 3, 4 }), false);
    _ = try program.createOutput("output", .float32, comptime .fromSlice(&.{ 2, 3, 4 }), false);

    const i = program.getInput("input");
    const o = program.getOutput("output");

    try std.testing.expect(i != null);
    try std.testing.expect(o != null);
}
