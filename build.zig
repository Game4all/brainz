const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addModule("brainz", .{
        .root_source_file = .{ .path = "src/root.zig" },
        .optimize = optimize,
        .target = target,
    });

    const lib_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/root.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);

    // examples

    const xor_example = b.addExecutable(.{
        .optimize = optimize,
        .target = target,
        .root_source_file = .{ .path = "examples/xor.zig" },
        .name = "xor_example",
    });

    xor_example.root_module.addImport("brainz", lib);

    var run_xor_example = b.addRunArtifact(xor_example);

    _ = b.step("example_xor", "Runs the XOR Mlp example")
        .dependOn(&run_xor_example.step);
}
