const perceptron = @import("perceptron.zig");

const std = @import("std");
pub const math = @import("math.zig");

// Common used activation functions.
pub const unit_step = math.unit_step;
pub const linear = math.linear;
pub const sigmoid = math.sigmoid;
pub const relu = math.relu;

pub const Perceptron = perceptron.Perceptron;
pub const DenseLayer = @import("nets/dense.zig").DenseLayer;
pub const Network = @import("nets/network.zig").Network;

comptime {
    std.testing.refAllDeclsRecursive(perceptron);
    std.testing.refAllDeclsRecursive(@import("nets/dense.zig"));
    std.testing.refAllDecls(math);
}
