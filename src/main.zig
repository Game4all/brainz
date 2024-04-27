const std = @import("std");
const aneurysm = @import("root.zig");

comptime {
    std.testing.refAllDeclsRecursive(aneurysm);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const alloc = gpa.allocator();

    const MLP = aneurysm.Network(@constCast(&[_]type{
        aneurysm.DenseLayer(2, 2, aneurysm.math.sigmoid),
        aneurysm.DenseLayer(2, 1, aneurysm.math.sigmoid),
    }));

    var net: MLP = .{};
    try net.init(alloc);
    defer net.deinit(alloc);

    const inputs = [_][2]f32{
        [2]f32{ 0.0, 0.0 },
        [2]f32{ 0.0, 1.0 },
        [2]f32{ 1.0, 0.0 },
        [2]f32{ 1.0, 1.0 },
    };

    const outputs = [_][1]f32{
        [1]f32{0.0},
        [1]f32{1.0},
        [1]f32{1.0},
        [1]f32{0.0},
    };

    for (0..1_500_000) |_| {
        for (inputs, outputs) |ins, outs| {
            _ = net.forward(@constCast(&ins));
            net.backwards(@constCast(&outs), 0.01);
        }
    }

    for (inputs, outputs) |ins, outs| {
        const prediction = net.forward(@constCast(&ins));
        std.log.info("Pred for {any} = {any} (real: {any})", .{ ins, prediction, outs });
    }
}
