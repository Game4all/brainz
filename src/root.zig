const std = @import("std");
const builtin = @import("builtin");

pub const activation = @import("activation.zig");

pub const tensor = @import("tensor.zig");
pub const ops = @import("ops.zig");

pub const Device = @import("device/Device.zig");

/// The default, preferred device to use on the target
pub const default_device = switch (builtin.target.cpu.arch) {
    .wasm32, .wasm64 => Device.DummyDevice,
    else => @import("device/CpuDevice.zig"),
};

pub const Dense = @import("dense.zig").Dense;

pub const Tensor = tensor.Tensor;
pub const TensorArena = tensor.Arena;

comptime {
    std.testing.refAllDeclsRecursive(tensor);
    std.testing.refAllDeclsRecursive(ops);
    std.testing.refAllDeclsRecursive(@import("dense.zig"));
}
