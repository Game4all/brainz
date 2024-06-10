const std = @import("std");

pub const activation = @import("activation.zig");
pub const loss = @import("loss.zig");

pub const matrix = @import("matrix.zig");
pub const ops = @import("matrix_ops.zig");

pub const Dense = @import("dense.zig").Dense;
pub const Matrix = matrix.Matrix;

comptime {
    std.testing.refAllDeclsRecursive(matrix);
    std.testing.refAllDeclsRecursive(ops);
    std.testing.refAllDeclsRecursive(@import("dense.zig"));
}
