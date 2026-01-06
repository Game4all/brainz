const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library compilation

    const lib = b.addModule("brainz", .{
        .root_source_file = b.path("src/root.zig"),
        .optimize = optimize,
        .target = target,
    });

    // Library tests

    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);

    // Library examples

    const examples = .{
        .{ "example_linear_regression", "examples/linear_regression.zig", "Linear regression" },
        .{ "example_xor", "examples/xor.zig", "XOR" },
    };

    inline for (examples) |example| {
        const name, const path, const description = example;

        const executable = b.addExecutable(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(path),
                .target = target,
                .optimize = optimize,
            }),
            .name = name,
        });

        executable.root_module.addImport("brainz", lib);
        const run_step = b.addRunArtifact(executable);
        _ = b.step(name, "Run the " ++ description ++ " example")
            .dependOn(&run_step.step);
    }
}
