const std = @import("std");

pub const activation = @import("activation.zig");
pub const loss = @import("loss.zig");

pub const tensor = @import("tensor.zig");
pub const ops = @import("ops.zig");

pub const Dense = @import("dense.zig").Dense;
pub const Tensor = tensor.Tensor;

comptime {
    std.testing.refAllDeclsRecursive(tensor);
    std.testing.refAllDeclsRecursive(ops);
    std.testing.refAllDeclsRecursive(@import("dense.zig"));
}
