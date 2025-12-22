const std = @import("std");
const tensor = @import("tensor.zig");

comptime {
    std.testing.refAllDeclsRecursive(tensor);
}
