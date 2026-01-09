const std = @import("std");

const tensor = @import("tensor.zig");
const op = @import("ops.zig");
const prog = @import("plan.zig");
const optimizers = @import("optim.zig");

pub const Tensor = tensor.Tensor;
pub const TensorArena = tensor.TensorArena;
pub const LinearPlan = prog.LinearPlan;
pub const ExecutionPlan = prog.ExecutionPlan;
pub const Dtype = tensor.Dtype;
pub const Shape = tensor.Shape;

// Reexported modules
pub const ops = op;
pub const optim = optimizers;
pub const nn = @import("nn.zig");

comptime {
    std.testing.refAllDeclsRecursive(tensor);
    std.testing.refAllDeclsRecursive(prog);
    std.testing.refAllDeclsRecursive(op);
    std.testing.refAllDeclsRecursive(nn);
    std.testing.refAllDeclsRecursive(optimizers);
}
