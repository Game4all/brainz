///! Represents a logical device capable of dispatching compute operations
const std = @import("std");
const Self = @This();

ptr: *anyopaque,
vtable: *const VTable,

/// Dispatches a computation to run on the device.
/// - `work` : The dispatch to run.
pub inline fn dispatch(self: *const @This(), work: Dispatch) !void {
    return self.vtable.dispatch(self.ptr, work);
}

/// Splits a dispatch into multiple sub-dispatches to be run.
/// - `work` : The dispatch to split into sub-chunks
/// - `num_pieces` : Number of sub-chunks to dispatch from the original dispatch.
pub inline fn dispatchChunks(self: *const @This(), work: Dispatch, num_pieces: usize) !void {
    return self.vtable.dispatchChunks(self.ptr, work, num_pieces);
}

/// Wait for the ongoing operations to finish before continuing further.
pub inline fn barrier(self: *const @This()) !void {
    return self.vtable.barrier(self.ptr);
}

pub const VTable = struct {
    /// Dispatches a computation to run on the device.
    dispatch: *const fn (ctx: *anyopaque, work: Dispatch) anyerror!void,
    /// Splits a dispatch into multiple sub-dispatches to be run.
    dispatchChunks: *const fn (ctx: *anyopaque, work: Dispatch, num_pieces: usize) anyerror!void,
    /// Wait for the ongoing operations to finish before continuing further.
    barrier: *const fn (ctx: *anyopaque) anyerror!void,
};

/// Represents a chunk of computation which can be run.
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
        fn dispatch(_: *anyopaque, work: Dispatch) !void {
            work.func(&work);
        }

        fn dispatchChunks(_: *anyopaque, work: Dispatch, num_pieces: usize) !void {
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
        .vtable = .{
            .dispatch = &vtable.dispatch,
            .dispatchChunks = &vtable.dispatchChunks,
            .barrier = &vtable.barrier,
        },
    };
};
