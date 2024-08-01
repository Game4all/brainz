///! A CPU multithreaded device implementation.
const std = @import("std");

const Allocator = std.mem.Allocator;
const Queue = std.fifo.LinearFifo;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const WaitGroup = std.Thread.WaitGroup;

const Device = @import("Device.zig");
const Dispatch = Device.Dispatch;

const Self = @This();

alloc: Allocator = undefined,

threads: []Thread = &[_]Thread{},
n_threads: usize = 0,
terminating: bool = false,

tasks_mutex: Mutex = .{},
tasks: Queue(Dispatch, .Slice) = Queue(Dispatch, .Slice).init(&[_]Dispatch{}),

sync_wg: WaitGroup = .{},

/// Initializes this device.
pub fn init(self: *Self, alloc: Allocator, n_th: ?usize) !void {
    self.alloc = alloc;

    const num_threads = n_th orelse std.Thread.getCpuCount() catch unreachable;

    const task_slice = try alloc.alloc(Dispatch, num_threads * 1024 * 4);
    self.tasks = Queue(Dispatch, .Slice).init(task_slice);
    errdefer self.deinit();

    self.threads = try alloc.alloc(Thread, num_threads);
    self.terminating = false;
    self.sync_wg.reset();

    for (self.threads) |*th| {
        th.* = try Thread.spawn(.{}, workerFunction, .{self});
        self.n_threads += 1;
    }
}

fn dispatch(ptr: *anyopaque, work: Dispatch, num_pieces: usize) !void {
    const self: *@This() = @ptrCast(@alignCast(ptr));
    self.tasks_mutex.lock();
    defer self.tasks_mutex.unlock();

    _ = self.sync_wg.state.fetchAdd(2 * num_pieces, .monotonic);

    for (0..num_pieces) |i| {
        var item = work;
        item.n_i = i;
        item.n_chunks = num_pieces;
        // self.sync_wg.start();

        try self.tasks.writeItem(item);
    }
}

fn barrier(ptr: *anyopaque) !void {
    const self: *@This() = @ptrCast(@alignCast(ptr));
    self.sync_wg.wait();
    self.sync_wg.reset();
}

pub fn device(self: *@This()) Device {
    return .{
        .ptr = @ptrCast(self),
        .vtable = &.{
            .dispatch = dispatch,
            .barrier = barrier,
        },
    };
}

/// Joins all threads and clears associated memory.
pub fn deinit(self: *Self) void {
    self.terminating = true;

    for (self.threads[0..self.n_threads]) |th|
        th.join();

    self.alloc.free(self.threads);
    self.alloc.free(self.tasks.buf);
}

fn workerFunction(self: *Self) void {
    while (true) {
        if (self.terminating)
            break;

        if (self.tasks_mutex.tryLock()) {
            if (self.tasks.readableLength() > 0) {
                const item = self.tasks.readItem() orelse unreachable;
                self.tasks_mutex.unlock();

                item.func(&item);

                self.sync_wg.finish();
            } else {
                self.tasks_mutex.unlock();
            }
        } else {
            std.Thread.yield() catch unreachable;
        }
    }
}
