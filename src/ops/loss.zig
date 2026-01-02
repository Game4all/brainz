const std = @import("std");
const tensor = @import("../tensor.zig");
const Tensor = tensor.Tensor;

pub fn forwardMSE(inputs: []const *const Tensor, output: *const Tensor, extra: ?*anyopaque) !void {
    _ = extra;
    const a = inputs[0]; // predictions
    const b = inputs[1]; // targets

    switch (output.dtype) {
        .float32 => {
            const out_data = output.slice(f32).?;
            const a_data = a.slice(f32).?;
            const b_data = b.slice(f32).?;
            const N: f32 = @floatFromInt(a.shape.totalLength());

            var sum: f32 = 0;
            for (a_data, b_data) |av, bv| {
                const diff = av - bv;
                sum += diff * diff;
            }

            // MSE = sum((a - b)^2) / N
            out_data[0] = sum / N;
        },
        .float64 => {
            const out_data = output.slice(f64).?;
            const a_data = a.slice(f64).?;
            const b_data = b.slice(f64).?;
            const N: f64 = @floatFromInt(a.shape.totalLength());

            var sum: f64 = 0;
            for (a_data, b_data) |av, bv| {
                const diff = av - bv;
                sum += diff * diff;
            }

            // MSE = sum((a - b)^2) / N
            out_data[0] = sum / N;
        },
        else => @panic("Unsupported dtype for MSE"),
    }
}

pub fn backwardMSE(inputs: []const *const Tensor, out: *const Tensor, grad_out: *const Tensor, extra: ?*anyopaque) !void {
    _ = out;
    _ = extra;
    const a = inputs[0]; // predictions
    const b = inputs[1]; // targets

    switch (grad_out.dtype) {
        .float32 => {
            const grad_val = grad_out.slice(f32).?[0];
            const N: f32 = @floatFromInt(a.shape.totalLength());
            const factor = (2.0 / N) * grad_val;

            const a_data = a.slice(f32).?;
            const b_data = b.slice(f32).?;

            if (a.requires_grad) {
                const a_grad = a.grad.?.slice(f32).?;
                for (0..a_data.len) |i| {
                    // dL/da = (2/N) * (a - b) * dL/dMSE
                    a_grad[i] += factor * (a_data[i] - b_data[i]);
                }
            }

            if (b.requires_grad) {
                const b_grad = b.grad.?.slice(f32).?;
                for (0..b_data.len) |i| {
                    // dL/db = (2/N) * (b - a) * dL/dMSE = - (2/N) * (a - b) * dL/dMSE
                    b_grad[i] += factor * (b_data[i] - a_data[i]);
                }
            }
        },
        .float64 => {
            const grad_val = grad_out.slice(f64).?[0];
            const N: f64 = @floatFromInt(a.shape.totalLength());
            const factor = (2.0 / N) * grad_val;

            const a_data = a.slice(f64).?;
            const b_data = b.slice(f64).?;

            if (a.requires_grad) {
                const a_grad = a.grad.?.slice(f64).?;
                for (0..a_data.len) |i| {
                    // dL/da = (2/N) * (a - b) * dL/dMSE
                    a_grad[i] += factor * (a_data[i] - b_data[i]);
                }
            }

            if (b.requires_grad) {
                const b_grad = b.grad.?.slice(f64).?;
                for (0..b_data.len) |i| {
                    // dL/db = (2/N) * (b - a) * dL/dMSE = - (2/N) * (a - b) * dL/dMSE
                    b_grad[i] += factor * (b_data[i] - a_data[i]);
                }
            }
        },
        else => @panic("Unsupported dtype for MSE backward"),
    }
}
