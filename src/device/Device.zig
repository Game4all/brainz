///! Represents a logical device capable of dispatching compute operations
const std = @import("std");
const Self = @This();

ptr: *anyopaque,
vtable: *const VTable,

/// Dispatches the specified operation to run.
/// - `work` : The dispatch to be run.
/// - `num_pieces` : Number of pieces which can be treated in parallel in this dispatch.
pub inline fn dispatch(self: *const @This(), work: Dispatch, num_pieces: usize) !void {
    return self.vtable.dispatch(self.ptr, work, num_pieces);
}

/// Wait for the ongoing operations to finish before continuing further.
pub inline fn barrier(self: *const @This()) !void {
    return self.vtable.barrier(self.ptr);
}

pub const VTable = struct {
    /// Dispatches the specified operation to run.
    dispatch: *const fn (ctx: *anyopaque, work: Dispatch, num_pieces: usize) anyerror!void,
    /// Wait for the ongoing operations to finish before continuing further.
    barrier: *const fn (ctx: *anyopaque) anyerror!void,
};

/// Represents a chunk of computation which can be run.
//TODO: this should not be part of the public API and be part of the internal device implementation.
pub const Dispatch = struct {
    pub const DispatchFunc = *const fn (item: *const @This()) void;
    /// The computation to perform.
    func: DispatchFunc,
    /// Total number of chunks in the computation if any.
    n_chunks: usize = 0,
    /// ID of the chunk to process (if there's multiple chunks).
    n_i: usize = 0,
    /// Arguments of this computation.
    args: [4]*anyopaque = undefined,
};

/// A dummy device implementation
pub const DummyDevice = blk: {
    const vtable = struct {
        fn dispatch(_: *anyopaque, work: Dispatch, num_pieces: usize) !void {
            for (0..num_pieces) |i| {
                var item = work;
                item.n_i = i;
                item.n_chunks = num_pieces;

                item.func(&item);
            }
        }

        fn barrier(_: *anyopaque) !void {}
    };

    break :blk Self{
        .ptr = undefined,
        .vtable = &.{
            .dispatch = &vtable.dispatch,
            .barrier = &vtable.barrier,
        },
    };
};
