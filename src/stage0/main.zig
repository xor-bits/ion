const std = @import("std");

const Tokenizer = @import("Tokenizer.zig");
const Parser = @import("Parser.zig");
// const IrGenerator = @import("IrGenerator.zig");
// const Sema = @import("Sema.zig");
const Codegen = @import("Codegen.zig");

pub fn main() !u8 {
    var gpf = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpf.deinit();
    const alloc = gpf.allocator();

    // const source_root = try std.process.getEnvMap(gpf.allocator());
    // source_root.get("");

    var args = try std.process.argsWithAllocator(alloc);
    defer args.deinit();

    _ = args.next().?;

    const source_path = args.next() orelse {
        std.log.err("expected source file argument", .{});
        return 1;
    };

    const destin_path = args.next() orelse {
        std.log.err("expected destination file argument", .{});
        return 1;
    };

    const source_file = try std.fs.cwd().openFile(source_path, .{});
    defer source_file.close();

    const destin_file = try std.fs.cwd().createFile(destin_path, .{});
    defer destin_file.close();

    var tokenizer: Tokenizer = .{ .source_file = source_file };
    defer tokenizer.deinit(alloc);
    try tokenizer.run(alloc);

    tokenizer.dump();

    var parser: Parser = .{ .tokenizer = &tokenizer };
    defer parser.deinit(alloc);
    try parser.run(alloc);

    parser.dump();

    // var ir_gen: IrGenerator = .{ .parser = &parser };
    // defer ir_gen.deinit(alloc);
    // try ir_gen.run(alloc);

    // ir_gen.dump();

    var codegen: Codegen = .{ .parser = &parser, .destin_file = destin_file };
    defer codegen.deinit(alloc);
    try codegen.run(alloc);

    parser.dump();

    return 0;
}

pub fn Range(
    comptime T: type,
    default: T,
) type {
    return struct {
        start: T = default,
        end: T = default,

        const Self = @This();

        pub fn len(
            self: Self,
        ) u32 {
            return self.end - self.start;
        }

        pub fn merge(
            a: Self,
            b: Self,
        ) Self {
            return .{
                .start = @min(a.start, b.start),
                .end = @max(a.end, b.end),
            };
        }

        pub fn read(
            self: Self,
            src: anytype,
        ) @TypeOf(src) {
            return src[self.start..self.end];
        }

        pub fn splitLast(
            self: Self,
        ) ?struct { Self, T } {
            if (self.len() == 0) return null;
            return .{
                .{
                    .start = self.start,
                    .end = self.end - 1,
                },
                self.end - 1,
            };
        }
    };
}

pub const NameHint = struct {
    prev: ?*const NameHint = null,
    part: []const u8 = "??",

    pub fn push(
        self: *const @This(),
        part: []const u8,
    ) @This() {
        return .{
            .prev = self,
            .part = part,
        };
    }

    pub fn generate(
        self: *const @This(),
        alloc: std.mem.Allocator,
    ) ![]const u8 {
        var len: usize = 0;
        var cur: ?*const @This() = self;
        while (cur) |next| {
            cur = next.prev;
            len += next.part.len + 1;
        }

        const name = try alloc.alloc(u8, len);
        cur = self;
        while (cur) |next| {
            cur = next.prev;
            name[len - 1] = '_';
            len -= next.part.len + 1;
            std.mem.copyForwards(u8, name[len..], next.part);
        }

        return name;
    }
};
