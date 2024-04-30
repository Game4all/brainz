const std = @import("std");
pub const math = @import("math.zig");
pub const activation = @import("nn/activation.zig");

pub const DenseLayer = @import("nn/dense.zig").DenseLayer;
pub const Network = @import("nn/network.zig").Network;

comptime {
    std.testing.refAllDeclsRecursive(@import("nn/dense.zig"));
    std.testing.refAllDeclsRecursive(@import("nn/network.zig"));
    std.testing.refAllDecls(math);
}
