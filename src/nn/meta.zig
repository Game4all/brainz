const std = @import("std");
const Activation = @import("activation.zig").Activation;

/// Information about a layer.
pub const LayerInfo = struct {
    /// Number of inputs this layer takes.
    num_inputs: usize,
    /// Number of outputs this layer takes.
    num_outputs: usize,
    /// Type of layer
    kind: [:0]const u8,
    /// Whether this layer has trainable parameters (weights / biases).
    has_weights: bool = false,
    /// Activation function this layer is using (if any).
    activation: ?Activation = null,
};

/// Returns information about the layer type or instance passed as an argument.
pub fn LayerInfoOf(layer: anytype) LayerInfo {
    switch (@typeInfo(@TypeOf(layer))) {
        .Type => {
            if (!@hasDecl(layer, "LAYER_INFO"))
                @compileError(std.fmt.comptimePrint("Layer {s} has no layer information attached.", .{@typeName(layer)}));

            const layer_info = @field(layer, "LAYER_INFO");

            if (@TypeOf(layer_info) != LayerInfo)
                @compileError(std.fmt.comptimePrint("Wrong type for `{s}.LAYER_INFO`. Expected {s} got {s}", .{ @typeName(layer), @typeName(LayerInfo), @typeName(@TypeOf(layer_info)) }));

            return @field(layer, "LAYER_INFO");
        },
        else => return LayerInfoOf(@TypeOf(layer)),
    }
}

/// Ensures the specified type adheres to the layer API.
pub fn Layer(layer: type) type {
    const layer_info = LayerInfoOf(layer);

    // layer base functions
    if (!std.meta.hasMethod(layer, "init"))
        @compileError("Missing `init` method in layer");

    if (!std.meta.hasMethod(layer, "deinit"))
        @compileError("Missing `deinit` method in layer");

    if (!std.meta.hasMethod(layer, "forward"))
        @compileError("Missing `forward` method in layer");

    // functions for layers with trainable parameters (weights and biases)
    if (layer_info.has_weights) {
        if (!std.meta.hasMethod(layer, "init_weights"))
            @compileError("Missing `init_weights` method in layer");

        if (!std.meta.hasMethod(layer, "backwards_out"))
            @compileError("Missing `backwards_out` method in layer");

        if (!std.meta.hasMethod(layer, "backwards"))
            @compileError("Missing `backwards` method in layer");

        if (!std.meta.hasMethod(layer, "update_weights"))
            @compileError("Missing `update_weights` method in layer");

        // actual weights for the layer.
        if (!@hasField(layer, "weights"))
            @compileError("Missing `weights` field in layer");

        // gradient values for backpropagation.
        if (!@hasField(layer, "grad"))
            @compileError("Missing `grad` field in layer");
    }

    return layer;
}
