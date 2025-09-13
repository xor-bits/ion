const std = @import("std");

pub fn main() !u8 {
    var gpf = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpf.deinit();

    // const source_root = try std.process.getEnvMap(gpf.allocator());
    // source_root.get("");

    var args = try std.process.argsWithAllocator(gpf.allocator());
    defer args.deinit();

    _ = args.next().?;

    const source_path = args.next() orelse {
        std.log.err("expected source file argument", .{});
        return 1;
    };

    const source = try std.fs.cwd().openFile(source_path, .{});
    defer source.close();

    var buffer: [0x8000]u8 = undefined;

    var source_reader = source.reader(&buffer);

    while (try next(&source_reader)) |ch| {
        std.debug.print("{c}", .{ch});
    }

    return 0;
}

fn next(reader: *std.fs.File.Reader) !?u8 {
    var ch: u8 = undefined;

    const count = reader.read(@ptrCast(&ch)) catch |err| switch (err) {
        error.EndOfStream => return null,
        error.ReadFailed => return err,
    };

    switch (count) {
        0 => return null,
        1 => return ch,
        else => unreachable,
    }
}
