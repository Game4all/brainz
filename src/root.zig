const std = @import("std");
const tensor = @import("tensor.zig");
const program = @import("program.zig");

const Tensor = tensor.Tensor;
const TensorArena = tensor.TensorArena;
const Dtype = tensor.Dtype;
const Shape = tensor.Shape;

comptime {
    std.testing.refAllDeclsRecursive(tensor);
    std.testing.refAllDeclsRecursive(program);
}
