const std = @import("std");
const brainz = @import("brainz");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const alloc = gpa.allocator();

    // Performs linear regression using a neural network.
    // by adding more layers with different activation functions, you could have the network imitate a more complex function.
    const RegNetwork = brainz.Network(@constCast(&[_]type{
        brainz.DenseLayer(1, 1, brainz.activation.Linear),
    }));

    var lin_reg: RegNetwork = .{};
    try lin_reg.init(alloc);

    const inputs = [_][1]f32{
        [1]f32{0.0},
        [1]f32{1.0},
        [1]f32{3.0},
        [1]f32{9.0},
    };

    // outputs follow f(x)=2x + 1
    const outputs = [_][1]f32{
        [1]f32{1.0},
        [1]f32{3.0},
        [1]f32{7.0},
        [1]f32{19.0},
    };

    for (0..5000) |_| {
        for (inputs, outputs) |in, out| {
            _ = lin_reg.forward(@constCast(&in));
            lin_reg.backwards(@constCast(&out), 0.001);
        }
    }

    for (inputs, outputs) |ins, outs| {
        const prediction = lin_reg.forward(@constCast(&ins));
        std.log.info("Pred for {any} = {any} (real: {any})", .{ ins, prediction, outs });
    }

    defer lin_reg.deinit(alloc);
}
