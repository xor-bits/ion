const std = @import("std");

//

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const stage0_compiler = b.addExecutable(.{
        .name = "ion-stage0",
        .root_module = b.createModule(.{
            .root_source_file = b.path("./src/stage0/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const install_stage0_compiler = b.addInstallArtifact(stage0_compiler, .{});
    b.default_step.dependOn(&install_stage0_compiler.step);

    const run_stage0_compiler = b.addRunArtifact(stage0_compiler);
    run_stage0_compiler.addFileArg(b.path("./src/stage1/main.ion"));

    const run_step = b.step("run", "run the compiler");
    run_step.dependOn(&run_stage0_compiler.step);
}
