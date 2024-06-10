const std = @import("std");

pub const activation = @import("nn/activation.zig");
pub const loss = @import("nn/loss.zig");
pub const meta = @import("nn/meta.zig");
pub const matrix = struct {
    usingnamespace @import("matrix.zig");
    usingnamespace @import("matrix_ops.zig");
};

pub const Matrix = matrix.Matrix;
pub const DenseLayer = @import("nn/dense.zig").DenseLayer;
pub const Network = @import("nn/network.zig").Network;

comptime {
    std.testing.refAllDeclsRecursive(@import("nn/dense.zig"));
    std.testing.refAllDeclsRecursive(@import("nn/network.zig"));
    std.testing.refAllDeclsRecursive(matrix);
}
