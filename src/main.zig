const std = @import("std");
const aneurysm = @import("root.zig");

comptime {
    std.testing.refAllDeclsRecursive(aneurysm);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const alloc = gpa.allocator();

    var arena = std.heap.ArenaAllocator.init(alloc);

    const XORMlp = aneurysm.Network(@constCast(&[_]type{
        aneurysm.DenseLayer(2, 3, aneurysm.sigmoid),
        aneurysm.DenseLayer(3, 1, aneurysm.sigmoid),
    }));

    var net: XORMlp = .{};

    try net.init(alloc);
    @memset(net.layers.@"0".biases, 0.0);
    @memset(net.layers.@"1".biases, 0.0);
    defer net.deinit(alloc);

    const INPUTS = [_][2]f32{
        [2]f32{ 0.0, 0.0 },
        [2]f32{ 0.0, 1.0 },
        [2]f32{ 1.0, 0.0 },
        [2]f32{ 1.0, 1.0 },
    };

    const OUTPUTS = [_][1]f32{
        [1]f32{0.0},
        [1]f32{1.0},
        [1]f32{1.0},
        [1]f32{0.0},
    };

    for (0..50_000) |i| {
        var e: f32 = 0.0;
        for (INPUTS, OUTPUTS) |inp, outp| {
            _ = net.forward(@constCast(&inp));
            e += try net.backprop(@constCast(&outp), @constCast(&inp), .{ .learn_rate = 0.5, .grad_clip_norm = 0.5 }, arena.allocator());
        }
        std.log.info("Iteration: {} | err={}", .{ i, e });
        _ = arena.reset(.retain_capacity);
    }

    for (INPUTS, OUTPUTS) |inp, outp| {
        const pred = net.forward(@constCast(&inp));
        std.log.info("Prediction for {any} => {any}  (Real is {any})", .{ inp, pred, outp });
    }

    arena.deinit();
}
