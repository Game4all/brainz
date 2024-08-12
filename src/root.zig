const std = @import("std");
const builtin = @import("builtin");

pub const activation = @import("activation.zig");

pub const tensor = @import("tensor.zig");
pub const ops = @import("ops.zig");

pub const Device = @import("device/Device.zig");

/// A dummy single-threaded device.
/// Available on all platforms.
pub const dummy_device = Device.DummyDevice;

/// The preferred high performance device type to use on the current compilation target.
pub const preferred_device_type = switch (builtin.target.cpu.arch) {
    .x86_64, .x86, .aarch64 => @import("device/CpuDevice.zig"),
    else => dummy_device,
};

pub const Dense = @import("dense.zig").Dense;

pub const Tensor = tensor.Tensor;
pub const TensorArena = tensor.Arena;

comptime {
    std.testing.refAllDeclsRecursive(tensor);
    std.testing.refAllDeclsRecursive(ops);
    std.testing.refAllDeclsRecursive(@import("dense.zig"));
}
