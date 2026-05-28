const std = @import("std");

const Tokenizer = @import("Tokenizer.zig");
const Parser = @import("Parser.zig");
const IrGenerator = @import("IrGenerator.zig");
const VirtualMachine = @import("VirtualMachine.zig");
// const Sema = @import("Sema.zig");
const Codegen = @import("Codegen.zig");

const Command = struct {
    self_exe: []const u8 = "",
    help: bool = false,
    dump_tokens: bool = false,
    dump_ast: bool = false,
    dump_ir: bool = false,
    dump_vm: bool = false,
    subcmd: ?SubCommand = null,

    const Flag = enum {
        help,
        @"dump-tokens",
        @"dump-ast",
        @"dump-ir",
        @"dump-vm",
    };
};

const SubCommand = union(enum) {
    eval,
    build: struct {
        source_path: []const u8 = "",
        destin_path: []const u8 = "",
    },

    const Tag = @typeInfo(@This()).@"union".tag_type.?;
};

pub fn main(init: std.process.Init) !u8 {
    const alloc = init.gpa;
    const io = init.io;

    // const source_root = try std.process.getEnvMap(gpf.allocator());
    // source_root.get("");

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const cli = parseCli(args) orelse {
        return 1;
    };

    const subcmd = cli.subcmd orelse {
        _ = help(cli.self_exe);
        return if (cli.help) 0 else 1;
    };

    return switch (subcmd) {
        .eval => eval(io, alloc, cli),
        .build => build(io, alloc, cli),
    };
}

fn parseCli(
    args: []const [:0]const u8,
) ?Command {
    var config: Command = .{};
    var subcmd_found: ?SubCommand.Tag = null;
    var nth_regular_arg: usize = 0;

    config.self_exe = args[0];
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "-h")) {
            config.help = true;
            continue;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            const flag = std.meta.stringToEnum(Command.Flag, arg[2..]) orelse {
                std.log.err("unknown cli flag '{s}'", .{arg[2..]});
                _ = help(config.self_exe);
                return null;
            };
            switch (flag) {
                .help => config.help = true,
                .@"dump-tokens" => config.dump_tokens = true,
                .@"dump-ast" => config.dump_ast = true,
                .@"dump-ir" => config.dump_ir = true,
                .@"dump-vm" => config.dump_vm = true,
            }
            continue;
        } else if (subcmd_found) |subcmd| {
            switch (subcmd) {
                .build => switch (nth_regular_arg) {
                    0 => {
                        config.subcmd.?.build.source_path = arg;
                        nth_regular_arg += 1;
                        continue;
                    },
                    1 => {
                        config.subcmd.?.build.destin_path = arg;
                        nth_regular_arg += 1;
                        continue;
                    },
                    else => {},
                },
                else => {},
            }
        } else if (std.meta.stringToEnum(SubCommand.Tag, arg)) |subcmd| {
            subcmd_found = subcmd;
            switch (subcmd) {
                .build => config.subcmd = .{ .build = .{} },
                .eval => config.subcmd = .eval,
            }
            continue;
        }

        std.log.err("unexpected cli argument '{s}'", .{arg});
        _ = help(config.self_exe);
        return null;
    }

    if (subcmd_found == null and !config.help) {
        _ = help(config.self_exe);
        return null;
    }

    if (subcmd_found == .build and nth_regular_arg == 0) {
        std.log.err("missing [source_path] argument", .{});
        _ = help(config.self_exe);
        return null;
    }

    if (subcmd_found == .build and nth_regular_arg == 1) {
        std.log.err("missing [output_path] argument", .{});
        _ = help(config.self_exe);
        return null;
    }

    return config;
}

fn help(
    self_exe: []const u8,
) u8 {
    std.debug.print(
        \\usage:
        \\  {s} [command] [options]
        \\
        \\commands
        \\  eval                : run ion code in a VM
        \\  build               : transpile ion code to zig code
        \\
        \\options
        \\  --help              : show this
        \\  --dump-tokens       : print tokens to stderr
        \\  --dump-ast          : print ast to stderr
        \\  --dump-ir           : print ir to stderr
        \\  --dump-vm           : print vm control flow to stderr
        \\
    , .{self_exe});
    return 0;
}

fn help_eval(
    self_exe: []const u8,
) u8 {
    std.debug.print(
        \\usage:
        \\  {s} eval [options]
        \\
        \\options
        \\  --help              : show this
        \\  --dump-tokens       : print tokens to stderr
        \\  --dump-ast          : print ast to stderr
        \\  --dump-ir           : print ir to stderr
        \\  --dump-vm           : print vm control flow to stderr
        \\
    , .{self_exe});
    return 0;
}

fn help_build(
    self_exe: []const u8,
) u8 {
    std.debug.print(
        \\usage:
        \\  {s} build [source_path] [output_path] [options]
        \\
        \\options
        \\  --help              : show this
        \\  --dump-tokens       : print tokens to stderr
        \\  --dump-ast          : print ast to stderr
        \\  --dump-ir           : print ir to stderr
        \\  --dump-vm           : print vm control flow to stderr
        \\
    , .{self_exe});
    return 0;
}

fn eval(
    io: std.Io,
    alloc: std.mem.Allocator,
    cli: Command,
) !u8 {
    if (cli.help) {
        return help_eval(cli.self_exe);
    }

    std.debug.print("write your program here, evaluate with Ctrl+D\n", .{});

    var stdin_buffer: [0x200]u8 = undefined;
    const stdin = std.Io.File.stdin();
    var stdin_reader = stdin.reader(io, &stdin_buffer);
    const source = try stdin_reader.interface.allocRemaining(
        alloc,
        .limited64(std.math.maxInt(u32)),
    );
    defer alloc.free(source);

    std.debug.print("\n", .{});

    // stdin.readPositionalAll(io, buffer: []u8, offset: u64)

    var tokenizer: Tokenizer = .{ .source = source };
    defer tokenizer.deinit(alloc);
    try tokenizer.run(alloc);

    if (cli.dump_tokens)
        tokenizer.dump();

    var parser: Parser = .{ .tokenizer = &tokenizer };
    defer parser.deinit(alloc);
    parser.run(alloc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            parser.printErrors();
            return 2;
        },
    };

    if (cli.dump_ast)
        parser.dump();

    var ir_gen: IrGenerator = .{ .parser = &parser };
    defer ir_gen.deinit(alloc);
    ir_gen.run(alloc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            ir_gen.printErrors();
            return 3;
        },
    };

    if (cli.dump_ir)
        ir_gen.dump();

    var vm: VirtualMachine = .{ .ir_gen = &ir_gen };
    vm.verbose = cli.dump_vm;
    // vm.gas = 1000;
    vm.mode = .eval;
    defer vm.deinit(alloc);
    vm.run(alloc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            vm.printErrors();
            return 4;
        },
    };

    if (cli.dump_vm)
        vm.dump();

    return 0;
}

fn build(
    io: std.Io,
    alloc: std.mem.Allocator,
    cli: Command,
) !u8 {
    if (cli.help) {
        return help_build(cli.self_exe);
    }

    const source_path = cli.subcmd.?.build.source_path;
    const destin_path = cli.subcmd.?.build.destin_path;

    const cwd = std.Io.Dir.cwd();

    const source_file = try cwd.openFile(io, source_path, .{});
    defer source_file.close(io);

    const destin_file = try cwd.createFile(io, destin_path, .{});
    defer destin_file.close(io);

    var source_buffer: [0x1000]u8 = undefined;
    var source_reader = source_file.reader(io, &source_buffer);
    const source = try source_reader.interface.allocRemaining(alloc, .limited64(std.math.maxInt(u32)));
    defer alloc.free(source);

    var output_buffer: [0x1000]u8 = undefined;
    var output_writer = destin_file.writer(io, &output_buffer);

    // var write_buffer: [0x8000]u8 = undefined;
    // var source_writer = self.destin_file.writer(&write_buffer);
    // const writer = &source_writer.interface;

    // std.debug.print("running lexer\n", .{});
    var tokenizer: Tokenizer = .{ .source = source };
    defer tokenizer.deinit(alloc);
    try tokenizer.run(alloc);

    if (cli.dump_tokens)
        tokenizer.dump();

    // std.debug.print("running parser\n", .{});
    var parser: Parser = .{ .tokenizer = &tokenizer };
    defer parser.deinit(alloc);
    parser.run(alloc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            parser.printErrors();
            return 2;
        },
    };

    if (cli.dump_ast)
        parser.dump();

    var ir_gen: IrGenerator = .{ .parser = &parser };
    defer ir_gen.deinit(alloc);
    ir_gen.run(alloc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            ir_gen.printErrors();
            return 3;
        },
    };

    if (cli.dump_ir)
        ir_gen.dump();

    var vm: VirtualMachine = .{ .ir_gen = &ir_gen };
    vm.verbose = cli.dump_vm;
    // vm.gas = 1000;
    vm.mode = .eval;
    defer vm.deinit(alloc);
    vm.run(alloc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            vm.printErrors();
            return 4;
        },
    };

    if (cli.dump_vm)
        vm.dump();

    // var sema: Sema = .{ .ir_gen = &ir_gen };
    // defer sema.deinit(alloc);
    // try sema.run(alloc);

    // sema.dump();

    // std.debug.print("running transpiler\n", .{});
    var codegen: Codegen = .{ .parser = &parser };
    defer codegen.deinit(alloc);
    codegen.run(alloc, &output_writer.interface) catch |err| switch (err) {
        error.OutOfMemory, error.WriteFailed => return err,
        else => {
            codegen.printErrors();
            return 3;
        },
    };

    try output_writer.flush();
    return 0;
}

pub fn Range(
    comptime T: type,
    default: T,
) type {
    return extern struct {
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

        pub fn expandLine(
            self: Self,
            src: []const u8,
        ) Self {
            std.debug.assert(self.start <= self.end);
            std.debug.assert(self.end <= src.len);

            var start = self.start;
            while (true) {
                if (start == 0) break;
                if (isNewline(src[start - 1])) break;
                start -= 1;
            }

            var end = self.end;
            while (true) {
                if (end == src.len) break;
                if (isNewline(src[end])) break;
                end += 1;
            }

            return .{ .start = start, .end = end };
        }

        pub fn findLineCol(
            self: Self,
            src: []const u8,
        ) [2]u32 {
            const before_this = src[0..self.start];

            var line_index: u32 = 0;
            var col_index: u32 = 0;

            for (before_this) |ch| {
                if (ch == '\n') {
                    line_index += 1;
                    col_index = 0;
                } else {
                    col_index += 1;
                }
            }

            return .{ line_index, col_index };
        }

        fn isNewline(
            c: u8,
        ) bool {
            return c == '\n' or c == '\r';
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
    prev: ?*const NameHint,
    part: []const u8,
    len: usize,

    pub fn new(
        base: []const u8,
    ) @This() {
        return .{
            .prev = null,
            .part = base,
            .len = base.len,
        };
    }

    pub fn push(
        self: *const @This(),
        part: []const u8,
    ) @This() {
        return .{
            .prev = self,
            .part = part,
            .len = self.len + part.len + 1,
        };
    }

    pub fn generate(
        self: *const @This(),
        alloc: std.mem.Allocator,
    ) error{OutOfMemory}![]const u8 {
        const name = try alloc.alloc(u8, self.len);
        @memset(name, '_');
        var cur: ?*const @This() = self;
        var n = self.len;
        while (cur) |next| {
            cur = next.prev;
            n -= next.part.len;
            std.mem.copyForwards(u8, name[n..], next.part);
            n -|= 1;
        }

        return name;
    }
};

pub const Kind = enum {
    err,

    pub fn str(self: @This()) []const u8 {
        return switch (self) {
            .err => "error",
        };
    }
};

pub fn Diagnostic(comptime Msg: type) type {
    return struct {
        src: struct {
            file: []const u8,
            code: []const u8,
            line: u32,
            col: u32,
            len: u32,

            pub fn fromSpan(
                span: Tokenizer.Span,
                src: []const u8,
            ) @This() {
                const code = span.expandLine(src).read(src);
                const line, const col = span.findLineCol(src);
                return .{
                    .file = "<root>",
                    .code = code,
                    .line = line,
                    .col = col,
                    .len = span.len(),
                };
            }
        },
        msg: Msg,
        kind: Kind,

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print("{s}:{}:{}: ", .{
                self.src.file,
                self.src.line + 1,
                self.src.col + 1,
            });
            try writer.print("{s}: {f}\n", .{
                self.kind.str(),
                self.msg,
            });
            try writer.writeAll(self.src.code);
            try writer.writeByte('\n');

            try writer.splatByteAll(' ', self.src.col);
            try writer.writeByte('^');
            try writer.splatByteAll('~', self.src.len -| 1);
        }
    };
}
