pub const BackpropParams = struct {
    /// The rate at which the network should learn.
    learn_rate: f32 = 0.005,
    /// The max norm that gradients can have without getting clipped.
    grad_clip_norm: f32 = 0.01,
};
