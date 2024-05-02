const std = @import("std");
const brainz = @import("brainz");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const alloc = gpa.allocator();

    const MLP = brainz.Network(@constCast(&[_]type{
        brainz.DenseLayer(4, 3, brainz.activation.Sigmoid),
        brainz.DenseLayer(3, 3, brainz.activation.Sigmoid),
        brainz.DenseLayer(3, 4, brainz.activation.Softmax),
    }));

    var net: MLP = .{};
    try net.init(alloc);
    defer net.deinit(alloc);

    const inputs = [_][4]f32{
        [4]f32{ 1.0, 0.0, 0.0, 0.0 },
        [4]f32{ 1.0, 1.0, 0.0, 0.0 },
        [4]f32{ 0.0, 1.0, 1.0, 0.0 },
        [4]f32{ 0.0, 0.0, 1.0, 1.0 },
        [4]f32{ 1.0, 0.0, 0.0, 1.0 },
    };

    const outputs = [_][4]f32{
        [4]f32{ 1.0, 0.0, 0.0, 0.0 },
        [4]f32{ 0.0, 1.0, 0.0, 0.0 },
        [4]f32{ 0.0, 0.0, 1.0, 0.0 },
        [4]f32{ 0.0, 0.0, 0.0, 1.0 },
        [4]f32{ 1.0, 0.0, 0.0, 1.0 },
    };

    for (0..1_500_000) |i| {
        var e: f32 = 0.0;
        for (inputs, outputs) |ins, outs| {
            _ = net.forward(@constCast(&ins));
            e += net.backwards(@constCast(&outs), 0.1, brainz.loss.MSE);
        }

        if (i % 10_000 == 0)
            std.log.info("loss: {}", .{e});
    }

    for (inputs, outputs) |ins, outs| {
        const prediction = net.forward(@constCast(&ins));
        std.log.info("Pred for {any} = {any} (real: {any})", .{ ins, brainz.math.argmax(prediction), brainz.math.argmax(@constCast(&outs)) });
    }
}
