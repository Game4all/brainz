const std = @import("std");
const builtin = @import("builtin");

pub const activation = @import("activation.zig");

pub const tensor = @import("tensor.zig");
pub const ops = @import("ops.zig");

pub const Device = @import("device/Device.zig");
pub const default_device = if (builtin.single_threaded) Device.DummyDevice else @import("device/CpuDevice.zig");

pub const Dense = @import("dense.zig").Dense;

pub const Tensor = tensor.Tensor;
pub const TensorArena = tensor.Arena;

comptime {
    std.testing.refAllDeclsRecursive(tensor);
    std.testing.refAllDeclsRecursive(ops);
    std.testing.refAllDeclsRecursive(@import("dense.zig"));
}
