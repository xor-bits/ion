const std = @import("std");

//

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const stage0 = b.createModule(.{
        .root_source_file = b.path("./src/stage0/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const stage0_compiler = b.addExecutable(.{
        .name = "ion-stage0",
        .root_module = stage0,
    });

    const install_stage0_compiler = b.addInstallArtifact(stage0_compiler, .{});
    b.default_step.dependOn(&install_stage0_compiler.step);

    const run_stage0_compiler = b.addRunArtifact(stage0_compiler);
    run_stage0_compiler.addFileArg(b.path("./src/stage1/main.ion"));
    const stage1_transpiled_src = run_stage0_compiler.addOutputFileArg("out.zig");

    const run_step = b.step("run", "run the compiler");

    if (b.option(bool, "dump", "dump stage0 output") == true) {
        const dump_stage1_transpiled_src = b.addSystemCommand(&.{
            "cat",
        });
        dump_stage1_transpiled_src.addFileArg(stage1_transpiled_src);
        run_step.dependOn(&dump_stage1_transpiled_src.step);
    }

    const stage1 = b.createModule(.{
        .root_source_file = stage1_transpiled_src,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    stage1.linkSystemLibrary("LLVM-21", .{});

    const stage1_compiler = b.addExecutable(.{
        .name = "ion-stage1",
        .root_module = stage1,
    });

    const run_stage1_compiler = b.addRunArtifact(stage1_compiler);
    run_step.dependOn(&run_stage1_compiler.step);
}
