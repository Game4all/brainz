const std = @import("std");
const builtin = @import("builtin");

/// Available activation functions.
pub const activation = @import("activation.zig");
/// Available tensor operations.
pub const ops = @import("ops.zig");

const tensor = @import("tensor.zig");

/// Represents a logical device capable of dispatching compute operations.
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

comptime {
    std.testing.refAllDeclsRecursive(tensor);
    std.testing.refAllDeclsRecursive(ops);
    std.testing.refAllDeclsRecursive(@import("dense.zig"));
}
