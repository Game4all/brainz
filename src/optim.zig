const std = @import("std");
const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;

/// A very simple stochastic optimizer
pub const SGD = struct {
    /// The parameters optimized by this optimizer
    parameters: []const *const Tensor,
    /// The learning rate
    lr: f32,

    pub inline fn init(params: []const *const Tensor, lr: f32) @This() {
        return .{
            .parameters = params,
            .lr = lr,
        };
    }

    /// Resets the gradients of all optimized parameters to zero.
    /// # Note
    /// - MUST be called before program.backward()
    pub fn zeroGrad(self: *SGD) void {
        for (self.parameters) |param| {
            if (param.grad) |grad| {
                switch (grad.dtype) {
                    .float32 => if (grad.slice(f32)) |s| @memset(s, 0),
                    .float64 => if (grad.slice(f64)) |s| @memset(s, 0),
                    else => {},
                }
            }
        }
    }

    /// Performs a single optimization step (updates parameters).
    pub fn step(self: *SGD) void {
        for (self.parameters) |param| {
            const grad = param.grad orelse continue;

            const has_storage = param.hasStorage();
            const grad_has_storage = grad.hasStorage();

            if (!has_storage or !grad_has_storage) continue;

            switch (param.dtype) {
                .float32 => self.stepImpl(f32, param, grad),
                .float64 => self.stepImpl(f64, param, grad),
                else => {},
            }
        }
    }

    fn stepImpl(self: *SGD, comptime T: type, param: *const Tensor, grad: *const Tensor) void {
        const p_data = param.slice(T).?;
        const g_data = grad.slice(T).?;
        const lr = @as(T, @floatCast(self.lr));

        for (p_data, g_data) |*p, g| {
            p.* -= lr * g;
        }
    }
};

test "Optimizer: SGD step and zeroGrad integration" {
    const testing = std.testing;
    const TensorArena = tensor.TensorArena;
    const Shape = tensor.Shape;

    var memArena = std.heap.ArenaAllocator.init(testing.allocator);
    defer memArena.deinit();

    var tensorArena: TensorArena = .init(memArena.allocator());
    defer tensorArena.deinit();

    const shape: Shape = comptime .fromSlice(&.{2});

    // we manually create and attach the gradient tensor to the original gradient because we don't use a program which would do it for us
    const param = try tensorArena.makeTensor(.float32, shape, true);
    const grad = try tensorArena.makeTensor(.float32, shape, false);

    // attaching the gradient tensor manually
    const mut_param: *tensor.Tensor = @constCast(param);
    mut_param.grad = grad;

    try tensorArena.allocateStorage();

    const p_data = param.slice(f32).?;
    @memcpy(p_data, &[_]f32{ 1.0, 2.0 });

    const g_data = grad.slice(f32).?;
    @memcpy(g_data, &[_]f32{ 0.5, 1.0 });

    var optim: SGD = SGD.init(&.{param}, 0.1);

    // to be expected:
    // p[0] = 1.0 - 0.1 * 0.5 = 0.95
    // p[1] = 2.0 - 0.1 * 1.0 = 1.9
    optim.step();

    try testing.expectApproxEqAbs(@as(f32, 0.95), p_data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.9), p_data[1], 1e-5);

    // testing gradients are zeroed out
    optim.zeroGrad();
    try testing.expectEqual(@as(f32, 0.0), g_data[0]);
    try testing.expectEqual(@as(f32, 0.0), g_data[1]);
}
