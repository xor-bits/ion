const std = @import("std");

const Tokenizer = @import("Tokenizer.zig");
const Parser = @import("Parser.zig");
const IrGenerator = @import("IrGenerator.zig");
// const Sema = @import("Sema.zig");
// const Codegen = @import("Codegen.zig");

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

    const source_file = try std.fs.cwd().openFile(source_path, .{});
    defer source_file.close();

    var read_buffer: [0x8000]u8 = undefined;

    var source_reader = source_file.reader(&read_buffer);
    const source_size = try source_reader.getSize();
    const source = try alloc.alloc(u8, source_size);
    defer alloc.free(source);
    std.debug.assert(try source_reader.read(source) == source_size);

    var tokenizer: Tokenizer = .{ .source = source };
    // while (tokenizer.next()) |_| {}

    var parser: Parser = .{ .tokenizer = &tokenizer };
    defer parser.deinit(alloc);
    try parser.run(alloc);

    parser.dump();

    var ir_gen: IrGenerator = .{ .parser = &parser };
    defer ir_gen.deinit(alloc);
    try ir_gen.run(alloc);

    // var parser: Parser = .init(alloc, tokenizer.tokens.items, tokenizer.spans.items, source);
    // defer parser.deinit();
    // try parser.run();

    // var analyzer: SemanticAnalyzer = .init(alloc, parser.nodes.items, source);
    // defer analyzer.deinit();
    // try analyzer.run();

    // // var ir_gen: IrGenerator = .init(alloc, parser.nodes.items, source);
    // // defer ir_gen.deinit();
    // // try ir_gen.run();

    // var codegen: Codegen = .init(alloc, parser.nodes.items, source);
    // defer codegen.deinit();
    // try codegen.run();

    // const ir_file = try std.fs.cwd().createFile("out.ll", .{});
    // defer ir_file.close();

    // var write_buffer: [0x8000]u8 = undefined;
    // var ir_writer = ir_file.writer(&write_buffer);
    // try codegen.dump(&ir_writer.interface);
    // try ir_writer.interface.flush();

    return 0;
}

pub fn Range(
    comptime T: type,
    default: T,
) type {
    return struct {
        start: T = default,
        end: T = default,

        pub fn len(
            self: @This(),
        ) u32 {
            return self.end - self.start;
        }

        pub fn merge(
            a: @This(),
            b: @This(),
        ) @This() {
            return .{
                .start = @min(a.start, b.start),
                .end = @max(a.end, b.end),
            };
        }

        pub fn read(
            self: @This(),
            src: anytype,
        ) @TypeOf(src) {
            return src[self.start..self.end];
        }
    };
}
