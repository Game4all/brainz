const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addModule("brainz", .{
        .root_source_file = b.path("src/root.zig"),
        .optimize = optimize,
        .target = target,
    });

    const lib_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);

    // examples

    const examples = .{
        .{ "example_xor", "examples/xor.zig", "XOR mlp" },
        .{ "example_linreg", "examples/linear_regression.zig", "Linear regression" },
        .{ "example_classification", "examples/classification.zig", "Classificator" },
    };

    inline for (examples) |example| {
        const name, const path, const description = example;

        const executable = b.addExecutable(.{
            .optimize = optimize,
            .target = target,
            .root_source_file = b.path(path),
            .name = name,
        });

        if (std.mem.eql(u8, name, "example_classification"))
            executable.addIncludePath(b.path("examples/datasets/"));

        executable.root_module.addImport("brainz", lib);
        const run_step = b.addRunArtifact(executable);
        _ = b.step(name, "Run the " ++ description ++ " example")
            .dependOn(&run_step.step);
    }
}
