<div align="center">
    <h1><code>brainz</code></h1>
    <i>Braaaaaaaaaaaaaiiiiiiiinnnnnnnnsssss 🧠🧟‍♂️</i>
    <br/>
    Simple Zig neural network library with no dependencies.
    <hr>
</div>

> **Warning**
> This is a work in progress.

## Getting started

```zig
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const alloc = gpa.allocator();

    // Declare a network and its layers.
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

    const outputs = [_][1]f32{
        [1]f32{1.0},
        [1]f32{3.0},
        [1]f32{7.0},
        [1]f32{19.0},
    };

    // train it for 5000 epochs
    for (0..5000) |i| {
        var e: f32 = 0.0;
        for (inputs, outputs) |in, out| {
            _ = lin_reg.forward(@constCast(&in));
            e += lin_reg.backwards(@constCast(&out), 0.001, brainz.loss.MSE);
        }

        if (i % 50 == 0)
            std.log.info("loss: {}", .{e});
    }

    for (inputs, outputs) |ins, outs| {
        const prediction = lin_reg.forward(@constCast(&ins));
        std.log.info("Pred for {any} = {any} (real: {any})", .{ ins, prediction, outs });
    }

    defer lin_reg.deinit(alloc);
}
```