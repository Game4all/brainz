const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;

fn dispatchMSEForward(comptime ty: type, pred: *const Tensor, target: *const Tensor, output: *const Tensor) void {
    const out_data = output.slice(ty).?;
    const a_data = pred.slice(ty).?;
    const b_data = target.slice(ty).?;
    const N: ty = @floatFromInt(pred.shape.totalLength());

    var sum: ty = 0;
    for (a_data, b_data) |av, bv| {
        const diff = av - bv;
        sum += diff * diff;
    }

    out_data[0] = sum / N;
}

pub fn forwardMSE(inputs: []const *const Tensor, output: *const Tensor, extra: ?*anyopaque) !void {
    _ = extra;
    const a = inputs[0]; // predictions
    const b = inputs[1]; // targets

    switch (output.dtype) {
        .float32 => dispatchMSEForward(f32, a, b, output),
        .float64 => dispatchMSEForward(f64, a, b, output),
        else => @panic("Unsupported dtype for MSE"),
    }
}

fn dispatchMSEBackward(comptime ty: type, pred: *const Tensor, target: *const Tensor, grad_out: *const Tensor) void {
    const a = pred;
    const b = target;
    const grad_val = grad_out.slice(ty).?[0];
    const N: ty = @floatFromInt(a.shape.totalLength());
    const factor = (2.0 / N) * grad_val;

    const a_data = a.slice(ty).?;
    const b_data = b.slice(ty).?;

    if (a.requires_grad) {
        const a_grad = a.grad.?.slice(ty).?;
        for (0..a_data.len) |i| {
            // dL/da = (2/N) * (a - b) * dL/dMSE
            a_grad[i] += factor * (a_data[i] - b_data[i]);
        }
    }

    if (b.requires_grad) {
        const b_grad = b.grad.?.slice(ty).?;
        for (0..b_data.len) |i| {
            // dL/db = (2/N) * (b - a) * dL/dMSE = - (2/N) * (a - b) * dL/dMSE
            b_grad[i] += factor * (b_data[i] - a_data[i]);
        }
    }
}

pub fn backwardMSE(inputs: []const *const Tensor, out: *const Tensor, grad_out: *const Tensor, extra: ?*anyopaque) !void {
    _ = out;
    _ = extra;
    const a = inputs[0]; // predictions
    const b = inputs[1]; // targets

    switch (grad_out.dtype) {
        .float32 => dispatchMSEBackward(f32, a, b, grad_out),
        .float64 => dispatchMSEBackward(f64, a, b, grad_out),
        else => @panic("Unsupported dtype for MSE backward"),
    }
}
