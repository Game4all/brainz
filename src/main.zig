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
        aneurysm.DenseLayer(2, 2, aneurysm.math.relu),
        aneurysm.DenseLayer(2, 1, aneurysm.math.sigmoid),
    }));

    var net: MLP = .{};
    try net.init(alloc);
    // try net.init_weights(@constCast(&WEIGHTS), @constCast(&BIASES), alloc);
    defer net.deinit(alloc);

    // const A = &[_]f32{ 2.0, -1 };
    // const B = &[_]f32{1.0};

    // for (0..1000) |_| {
    //     _ = net.forward(@constCast(A));
    //     net.backwards(@constCast(A), @constCast(B), 0.1);
    // }

    // std.log.info("Pred: {any}", .{net.forward(@constCast(A))});

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
        // try std.testing.expect(std.mem.eql(f32, prediction, &outs));
        std.log.info("Pred for {any} = {any} (real: {any})", .{ ins, prediction, outs });
    }

    // std.log.info("{any}", .{net.forward(@constCast(A))});

    // const B = &[_]f32{1.0};

}
