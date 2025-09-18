const std = @import("std");

// const llvm = @cImport({
//     @cInclude("llvm-c/Core.h");
//     @cInclude("llvm-c/Analysis.h");
// });

pub fn main() !u8 {

    // _ = llvm.LLVMCreateBuilder();
    // const mod = llvm.LLVMModuleCreateWithName("ion");
    // const main_fn_ty = llvm.LLVMFunctionType(llvm.LLVMInt32Type(), null, 0, 0);
    // _ = llvm.LLVMAddFunction(mod, "main", main_fn_ty);
    // var err: [*c]u8 = null;
    // _ = llvm.LLVMVerifyModule(mod, llvm.LLVMAbortProcessAction, &err);
    // llvm.LLVMDumpModule(mod);

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

    var tokenizer: Tokenizer = .init(alloc, source);
    defer tokenizer.deinit();
    try tokenizer.run();

    var parser: Parser = .init(alloc, tokenizer.tokens.items, tokenizer.spans.items, source);
    defer parser.deinit();
    try parser.run();

    var analyzer: SemanticAnalyzer = .init(alloc, parser.nodes.items, source);
    defer analyzer.deinit();
    try analyzer.run();

    // var ir_gen: IrGenerator = .init(alloc, parser.nodes.items, source);
    // defer ir_gen.deinit();
    // try ir_gen.run();

    var codegen: Codegen = .init(alloc, parser.nodes.items, source);
    defer codegen.deinit();
    try codegen.run();

    const ir_file = try std.fs.cwd().createFile("out.ll", .{});
    defer ir_file.close();

    var write_buffer: [0x8000]u8 = undefined;
    var ir_writer = ir_file.writer(&write_buffer);
    try codegen.dump(&ir_writer.interface);
    try ir_writer.interface.flush();

    return 0;
}

pub fn Range(comptime T: type, default: T) type {
    return struct {
        start: T = default,
        end: T = default,

        pub fn len(self: @This()) u32 {
            return self.end - self.start;
        }

        pub fn merge(a: @This(), b: @This()) @This() {
            return .{
                .start = @min(a.start, b.start),
                .end = @max(a.end, b.end),
            };
        }
    };
}

pub const Token = enum {
    ident,
    line_comment,
    block_comment,

    let,
    mut,
    @"fn",

    semi,
    colon,
    comma,
    dot,
    double_dot,
    lparen,
    rparen,
    lbrace,
    rbrace,
    lbracket,
    rbracket,

    eq,
    neq,
    lt,
    le,
    gt,
    ge,

    exclam,
    plus,
    minus,
    asterisk,
    slash,
    percent,

    str_lit,
    char_lit,
    int_lit,
    float_lit,
};

pub const Span = Range(u32, 0);

pub const Tokenizer = struct {
    tokens: std.ArrayListUnmanaged(Token) = .{},
    spans: std.ArrayListUnmanaged(Span) = .{},
    alloc: std.mem.Allocator,

    source: []const u8,
    cursor: u32 = 0,

    pub fn init(alloc: std.mem.Allocator, source: []const u8) @This() {
        return .{ .alloc = alloc, .source = source };
    }

    pub fn deinit(self: *@This()) void {
        self.spans.deinit(self.alloc);
        self.tokens.deinit(self.alloc);
    }

    pub fn run(self: *@This()) !void {
        // assume approx one token per 3 characters
        try self.tokens.ensureTotalCapacity(self.alloc, self.source.len / 3);
        try self.spans.ensureTotalCapacity(self.alloc, self.source.len / 3);
        self.tokens.clearRetainingCapacity();
        self.spans.clearRetainingCapacity();

        while (try self.next()) |result| {
            const tok, const span = result;
            std.debug.print("{s:>16}: `{s}`\n", .{
                @tagName(tok),
                self.readSpan(span),
            });

            try self.tokens.append(self.alloc, tok);
            try self.spans.append(self.alloc, span);
        }
    }

    fn popIf(self: *@This(), comptime pred: fn (u8) bool) ?struct { u8, Span } {
        const ch, const span = self.peek() orelse return null;
        if (pred(ch)) {
            self.advance();
            return .{ ch, span };
        } else {
            return null;
        }
    }

    fn popIfEql(self: *@This(), expect: u8) ?struct { u8, Span } {
        const ch, const span = self.peek() orelse return null;
        if (ch == expect) {
            self.advance();
            return .{ ch, span };
        } else {
            return null;
        }
    }

    fn pop(self: *@This()) ?struct { u8, Span } {
        defer self.advance();
        return self.peek();
    }

    fn peek(self: *@This()) ?struct { u8, Span } {
        if (self.cursor >= self.source.len) return null;
        return .{
            self.source[self.cursor],
            Span{ .start = self.cursor, .end = self.cursor + 1 },
        };
    }

    fn advance(self: *@This()) void {
        if (self.cursor >= self.source.len) return;
        self.cursor += 1;
    }

    fn readSpan(self: *@This(), span: Span) []const u8 {
        return self.source[span.start..span.end];
    }

    pub fn next(self: *@This()) !?struct { Token, Span } {
        var ch, var start_span = self.pop() orelse return null;
        var span = start_span;

        const State = enum {
            start,

            comment,
            line_comment,
            block_comment,

            ident,

            str_lit,
            char_lit,
            num_lit,
        };

        loop: switch (State.start) {
            .start => switch (ch) {
                '/' => {
                    ch, span = self.pop() orelse return null;
                    continue :loop State.comment;
                },

                'a'...'z',
                'A'...'Z',
                => {
                    continue :loop State.ident;
                },

                '"' => {
                    ch, span = self.pop() orelse return null;
                    continue :loop State.str_lit;
                },
                '\'' => {
                    ch, span = self.pop() orelse return null;
                    continue :loop State.char_lit;
                },

                else => {
                    if (std.ascii.isDigit(ch)) {
                        continue :loop State.num_lit;
                    }

                    if (std.ascii.isWhitespace(ch)) {
                        // ignore whitespace
                        ch, span = self.pop() orelse return null;
                        start_span = span;
                        continue :loop State.start;
                    }

                    const simple_tok: Token = switch (ch) {
                        ';' => .semi,
                        ':' => .colon,
                        ',' => .comma,
                        '.' => b: {
                            ch, span = self.popIfEql('.') orelse break :b .dot;
                            break :b .double_dot;
                        },
                        '(' => .lparen,
                        ')' => .rparen,
                        '{' => .lbrace,
                        '}' => .rbrace,
                        '[' => .lbracket,
                        ']' => .rbracket,
                        '=' => .eq,
                        '!' => b: {
                            ch, span = self.popIfEql('=') orelse break :b .exclam;
                            break :b .neq;
                        },
                        '<' => b: {
                            ch, span = self.popIfEql('=') orelse break :b .lt;
                            break :b .le;
                        },
                        '>' => b: {
                            ch, span = self.popIfEql('=') orelse break :b .gt;
                            break :b .ge;
                        },
                        '+' => .plus,
                        '-' => .minus,
                        '*' => .asterisk,
                        '/' => .slash,
                        '%' => .percent,

                        else => {
                            std.debug.print("unexpected byte: '{s}'\n", .{
                                @as([]const u8, @ptrCast(&ch)),
                            });
                            return error.UnexpectedByte;
                        },
                    };
                    const full_span = start_span.merge(span);

                    return .{
                        simple_tok,
                        full_span,
                    };
                },
            },
            .comment => switch (ch) {
                '/' => continue :loop State.line_comment,
                '*' => continue :loop State.block_comment,
                else => {
                    if (std.ascii.isWhitespace(ch)) return .{
                        // just a `/` token
                        Token.slash,
                        start_span.merge(span),
                    };

                    std.debug.print("unexpected bytes: '{s}'\n", .{
                        self.readSpan(start_span.merge(span)),
                    });
                    return error.UnexpectedByte;
                },
            },
            .line_comment => {
                // ignore line comments
                while (ch != '\n')
                    ch, span = self.pop() orelse return null;
                while (true)
                    ch, span = self.popIfEql('\n') orelse break;
                start_span = span;
                continue :loop State.start;
            },
            .block_comment => {
                // ignore block comments
                var prev: u8 = ' ';
                while (true) {
                    defer prev = ch;

                    ch, span = self.pop() orelse return null;
                    if (prev == '*' and ch == '/') break;
                }

                ch, span = self.pop() orelse return null;
                start_span = span;
                continue :loop State.start;
            },
            .ident => {
                while (true) {
                    ch, span = self.popIf(std.ascii.isAlphanumeric) orelse break;
                }

                const full_span = start_span.merge(span);
                const ident = self.readSpan(full_span);

                const Keyword = enum {
                    let,
                    mut,
                    @"fn",
                };

                const keyword = std.meta.stringToEnum(Keyword, ident) orelse {
                    return .{
                        Token.ident,
                        full_span,
                    };
                };

                switch (keyword) {
                    inline else => |kw| {
                        return .{
                            @field(Token, @tagName(kw)),
                            full_span,
                        };
                    },
                }
            },
            .str_lit => {
                while (true) {
                    ch, span = self.popIf(struct {
                        fn pred(_ch: u8) bool {
                            return _ch != '"';
                        }
                    }.pred) orelse break;
                }
                ch, span = self.pop() orelse return null;

                return .{
                    Token.str_lit,
                    start_span.merge(span),
                };
            },
            .char_lit => {
                _ = self.pop() orelse return null;
                ch, span = self.pop() orelse return null;

                return .{
                    Token.char_lit,
                    start_span.merge(span),
                };
            },
            .num_lit => {
                var dot_found = false;
                while (true) {
                    const result = self.peek() orelse break;

                    if (dot_found and result.@"0" == '.') break;
                    if (result.@"0" == '.') {
                        dot_found = true;
                    } else if (!std.ascii.isAlphanumeric(result.@"0")) break;
                    ch, span = result;
                    self.advance();
                }

                const full_span = start_span.merge(span);
                // var str = self.readSpan(full_span);
                // var base: u8 = 10;

                // if (str.len >= 2) switch (str[1]) {
                //     'x' => {
                //         base = 16;
                //         str = str[2..];
                //     },
                //     'o' => {
                //         base = 8;
                //         str = str[2..];
                //     },
                //     'b' => {
                //         base = 2;
                //         str = str[2..];
                //     },
                //     else => {},
                // };

                if (dot_found) {
                    // var num: f128 = 0.0;
                    // var past_dot = false;
                    // var decimal: f128 = 0.0;
                    // for (str) |digit_ch| {
                    //     if (digit_ch == '.') {
                    //         past_dot = true;
                    //         continue;
                    //     }

                    //     const digit = std.fmt.charToDigit(digit_ch, base) catch {
                    //         std.debug.print("invalid digit in a float literal: '{s}'\n", .{
                    //             @as([]const u8, @ptrCast(&digit_ch)),
                    //         });
                    //         return null;
                    //     };

                    //     if (past_dot) {
                    //         decimal -= 1.0;
                    //         num += @as(f128, @floatFromInt(digit)) * std.math.pow(f128, 10.0, decimal);
                    //     } else {
                    //         num *= 10.0;
                    //         num += @floatFromInt(digit);
                    //     }
                    // }

                    return .{
                        Token.float_lit,
                        full_span,
                    };
                } else {
                    // var num: u128 = 0;
                    // for (str) |digit_ch| {
                    //     num *= 10;
                    //     num += digit_ch;
                    // }
                    return .{
                        Token.int_lit,
                        full_span,
                    };
                }
            },
        }
    }
};

pub const Node = union(enum) {
    @"struct": struct {
        // struct_kw: Span,
        // lbrace_kw: Span,
        contents: NodeId,
        // rbrace_kw: Span,
    },
    struct_contents: struct {
        fields: NodeRange,
        decls: NodeRange,
    },
    field: struct {
        ident: Span,
        // colon: Span,
        type: NodeId,
        // eq: Span,
        default: ?NodeId,
        // comma: Span,
    },
    decl: struct {
        // let: Span,
        // mut: ?Span,
        mut: bool,
        ident: Span,
        // eq: Span,
        expr: NodeId,
        // semi: Span,
    },
    @"fn": struct {
        // @"fn": Span,
        // lparen: Span,
        params: NodeRange,
        // rparen: Span,
        scope: NodeId,
    },
    scope: struct {
        // lbrace: Span,
        stmts: NodeRange,
        has_trailing_semi: bool,
        // rbrace: Span,
    },
    param: struct {
        ident: Span,
        // colon: Span,
        type: NodeId,
        // comma: Span,
    },
    binary_op: struct {
        lhs: NodeId,
        op: BinaryOp,
        rhs: NodeId,
    },
    unary_op: struct {
        op: UnaryOp,
        val: NodeId,
    },
    field_acc: struct {
        val: NodeId,
        ident: Span,
    },
    index_acc: struct {
        val: NodeId,
        // lbracket: Span,
        expr: NodeId,
        // rbracket: Span,
    },
    call: struct {
        val: NodeId,
        args: NodeRange,
    },
    access: struct {
        ident: Span,
    },
    str_lit: struct {
        tok: Span,
    },
    char_lit: struct {
        tok: Span,
        val: u8,
    },
    int_lit: struct {
        // tok: Span,
        val: u64,
    },
    float_lit: struct {
        // tok: Span,
        val: f64,
    },
};

pub const BinaryOp = enum {
    add,
    sub,
    mul,
    div,
    rem,
};

pub const UnaryOp = enum {
    neg,
    not,
};

pub const NodeId = u32;

pub const NodeRange = Range(NodeId, 0);

pub const Parser = struct {
    nodes: std.ArrayListUnmanaged(Node) = .{},
    alloc: std.mem.Allocator,

    tokens: []const Token,
    spans: []const Span,
    source: []const u8,
    cursor: u32 = 0,

    pub const Error = error{
        InvalidSyntax,
        TooManyAstNodes,
        OutOfMemory,
        EndOfFile,
    };

    pub fn init(
        alloc: std.mem.Allocator,
        tokens: []const Token,
        spans: []const Span,
        source: []const u8,
    ) @This() {
        return .{
            .alloc = alloc,
            .tokens = tokens,
            .spans = spans,
            .source = source,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.nodes.deinit(self.alloc);
    }

    pub fn run(self: *@This()) !void {
        std.debug.assert(self.spans.len == self.tokens.len);

        // assume approx one token per 3 characters
        try self.nodes.ensureTotalCapacity(self.alloc, self.source.len / 3);
        self.nodes.clearRetainingCapacity();

        const root = try self.allocNode(.{ .struct_contents = undefined }); // reserve idx 0 as the ast root
        std.debug.assert(root == 0);
        self.nodes.items[0] = try self.parseFile();

        // for (self.nodes.items, 0..) |node, i|
        //     std.debug.print("id={} node={}\n", .{ i, node });
        self.print(0, 0);
    }

    fn indent(depth: usize) void {
        for (0..depth) |_| std.debug.print("| ", .{});
    }

    pub fn print(self: *@This(), node: NodeId, depth: usize) void {
        indent(depth);
        // std.debug.print("id={} ", .{node});

        switch (self.nodes.items[node]) {
            .@"struct" => |v| {
                std.debug.print("struct:\n", .{});
                self.print(v.contents, depth + 1);
            },
            .struct_contents => |v| {
                std.debug.print("struct_contents:\n", .{});
                for (v.fields.start..v.fields.end) |i|
                    self.print(@truncate(i), depth + 1);
                for (v.decls.start..v.decls.end) |i|
                    self.print(@truncate(i), depth + 1);
            },
            .field => |v| {
                std.debug.print("field name='{s}':\n", .{
                    self.readSpan(v.ident),
                });
                self.print(v.type, depth + 1);
                if (v.default) |default|
                    self.print(default, depth + 1);
            },
            .decl => |v| {
                std.debug.print("decl mut={} name='{s}':\n", .{
                    v.mut,
                    self.readSpan(v.ident),
                });
                self.print(v.expr, depth + 1);
            },
            .@"fn" => |v| {
                std.debug.print("fn:\n", .{});
                indent(depth + 1);
                std.debug.print("params:\n", .{});
                for (v.params.start..v.params.end) |i|
                    self.print(@truncate(i), depth + 2);
                self.print(v.scope, depth + 1);
            },
            .scope => |v| {
                std.debug.print("scope autoreturn={}:\n", .{
                    !v.has_trailing_semi,
                });
                for (v.stmts.start..v.stmts.end) |i|
                    self.print(@truncate(i), depth + 1);
            },
            .param => |v| {
                std.debug.print("param name='{s}':\n", .{
                    self.readSpan(v.ident),
                });
                self.print(v.type, depth + 1);
            },
            .binary_op => |v| {
                std.debug.print("binary_op op={t}:\n", .{
                    v.op,
                });
                self.print(v.lhs, depth + 1);
                self.print(v.rhs, depth + 1);
            },
            .unary_op => |v| {
                std.debug.print("unary_op op={t}:\n", .{
                    v.op,
                });
                self.print(v.val, depth + 1);
            },
            .field_acc => |v| {
                std.debug.print("field_acc name='{s}':\n", .{
                    self.readSpan(v.ident),
                });
                self.print(v.val, depth + 1);
            },
            .index_acc => |v| {
                std.debug.print("index_acc:\n", .{});
                self.print(v.val, depth + 1);
                self.print(v.expr, depth + 1);
            },
            .call => |v| {
                std.debug.print("call:\n", .{});
                self.print(v.val, depth + 1);
                indent(depth + 1);
                std.debug.print("args:\n", .{});
                for (v.args.start..v.args.end) |i|
                    self.print(@truncate(i), depth + 2);
            },
            .access => |v| {
                std.debug.print("access: {s}\n", .{
                    self.readSpan(v.ident),
                });
            },
            .str_lit => |v| {
                std.debug.print("str_lit: {s}\n", .{
                    self.readSpan(v.tok),
                });
            },
            .char_lit => |v| {
                std.debug.print("char_lit raw={}: {s}\n", .{
                    v.val,
                    self.readSpan(v.tok),
                });
            },
            .int_lit => |v| {
                std.debug.print("int_lit: {d}\n", .{
                    v.val,
                });
            },
            .float_lit => |v| {
                std.debug.print("float_lit: {d}\n", .{
                    v.val,
                });
            },
        }
    }

    fn allocNode(self: *@This(), node: Node) Error!NodeId {
        // std.log.debug("alloc node={any}", .{node});
        const id = std.math.cast(u32, self.nodes.items.len) orelse
            return error.TooManyAstNodes;
        try self.nodes.append(self.alloc, node);
        return id;
    }

    fn allocNodes(self: *@This(), nodes: []const Node) Error!NodeRange {
        // std.log.debug("alloc nodes={any}", .{nodes});
        const start = std.math.cast(u32, self.nodes.items.len) orelse
            return error.TooManyAstNodes;
        const end = std.math.cast(u32, self.nodes.items.len + nodes.len) orelse
            return error.TooManyAstNodes;

        const slot = try self.nodes.addManyAsSlice(self.alloc, nodes.len);
        @memcpy(slot, nodes);
        return .{ .start = start, .end = end };
    }

    fn peekNth(self: *@This(), n: usize) Error!struct { Token, Span } {
        if (self.cursor + n >= self.tokens.len) return error.EndOfFile;
        return .{
            self.tokens[self.cursor + n],
            self.spans[self.cursor + n],
        };
    }

    fn peekNthToken(self: *@This(), n: usize) Error!Token {
        const tok, _ = try self.peekNth(n);
        return tok;
    }

    // fn expectNth(self: *@This(), n: usize, expect: Token) ?struct { Token, Span } {
    //     const tok, const span = self.peekNth(n) orelse return null;
    //     if (tok == expect) {
    //         return .{ tok, span };
    //     } else {
    //         return null;
    //     }
    // }

    // fn popIf(self: *@This(), comptime pred: fn (Token, Span) bool) ?struct { Token, Span } {
    //     const tok, const span = self.peekNth(0) catch return null;
    //     if (pred(tok, span)) {
    //         self.advance();
    //         return .{ tok, span };
    //     } else {
    //         return null;
    //     }
    // }

    fn popIfEql(self: *@This(), expect: Token) ?struct { Token, Span } {
        const tok, const span = self.peekNth(0) catch return null;
        if (tok == expect) {
            self.advance();
            return .{ tok, span };
        } else {
            return null;
        }
    }

    fn advance(self: *@This()) void {
        if (self.cursor >= self.tokens.len) return;
        self.cursor += 1;
    }

    fn readSpan(self: *@This(), span: Span) []const u8 {
        return self.source[span.start..span.end];
    }

    fn parseToken(self: *@This(), expect: Token) Error!Span {
        _, const span = self.popIfEql(expect) orelse
            return error.InvalidSyntax;
        return span;
    }

    fn parseFile(self: *@This()) Error!Node {
        // std.log.debug("parse file", .{});
        return self.parseStructContents();
    }

    fn parseStructContents(self: *@This()) Error!Node {
        // std.log.debug("parse struct contents", .{});
        var fields: std.ArrayList(Node) = .{};
        defer fields.deinit(self.alloc);
        var decls: std.ArrayList(Node) = .{};
        defer decls.deinit(self.alloc);

        while (true) {
            const tok = self.peekNthToken(0) catch break;
            switch (tok) {
                .ident => try fields.append(self.alloc, try self.parseField()),
                .let => try decls.append(self.alloc, try self.parseDecl()),
                .rbrace => break,
                else => return error.InvalidSyntax,
            }
        }

        return .{ .struct_contents = .{
            .fields = try self.allocNodes(fields.items),
            .decls = try self.allocNodes(decls.items),
        } };
    }

    fn parseField(self: *@This()) Error!Node {
        // std.log.debug("parse field", .{});
        const ident = try self.parseToken(.ident);
        const colon = try self.parseToken(.colon);
        const ty = try self.parseType();

        const tok = try self.peekNthToken(0);
        switch (tok) {
            .eq => {
                const eq = try self.parseToken(.eq);
                const default = try self.parseExpr();
                const comma = try self.parseToken(.comma);

                _ = .{ colon, eq, comma };

                return .{ .field = .{
                    .ident = ident,
                    .type = try self.allocNode(ty),
                    .default = try self.allocNode(default),
                } };
            },
            .comma => {
                const comma = try self.parseToken(.comma);

                _ = .{ colon, comma };

                return .{ .field = .{
                    .ident = ident,
                    .type = try self.allocNode(ty),
                    .default = null,
                } };
            },
            else => return error.InvalidSyntax,
        }
    }

    fn parseDecl(self: *@This()) Error!Node {
        // std.log.debug("parse decl", .{});
        const let = try self.parseToken(.let);
        var mut: ?Span = null;
        if (try self.peekNthToken(0) == .mut)
            mut = try self.parseToken(.mut);

        const ident = try self.parseToken(.ident);
        const eq = try self.parseToken(.eq);
        const expr = try self.parseExpr();
        const semi = try self.parseToken(.semi);

        _ = .{ let, eq, semi };

        return .{ .decl = .{
            .mut = mut != null,
            .ident = ident,
            .expr = try self.allocNode(expr),
        } };
    }

    fn parseType(_: *@This()) Error!Node {
        // std.log.debug("parse type", .{});
        std.debug.panic("todo", .{});
    }

    fn parseExpr(self: *@This()) Error!Node {
        // std.log.debug("parse expr", .{});
        var lhs = try self.parseTerm();

        while (true) {
            const op = switch (try self.peekNthToken(0)) {
                .asterisk => BinaryOp.mul,
                .slash => BinaryOp.div,
                .percent => BinaryOp.rem,
                else => break,
            };
            _ = self.advance();
            const rhs = try self.parseTerm();

            const prev = try self.allocNode(lhs); // https://github.com/ziglang/zig/issues/24627
            lhs = .{ .binary_op = .{
                .lhs = prev,
                .op = op,
                .rhs = try self.allocNode(rhs),
            } };
        }

        return lhs;
    }

    fn parseTerm(self: *@This()) Error!Node {
        // std.log.debug("parse term", .{});
        var lhs = try self.parseFactor();

        while (true) {
            const op = switch (try self.peekNthToken(0)) {
                .plus => BinaryOp.add,
                .minus => BinaryOp.sub,
                else => break,
            };
            _ = self.advance();
            const rhs = try self.parseFactor();

            const prev = try self.allocNode(lhs); // https://github.com/ziglang/zig/issues/24627
            lhs = .{ .binary_op = .{
                .lhs = prev,
                .op = op,
                .rhs = try self.allocNode(rhs),
            } };
        }

        return lhs;
    }

    fn parseFactor(self: *@This()) Error!Node {
        // std.log.debug("parse factor", .{});
        const op = switch (try self.peekNthToken(0)) {
            .plus => {
                // 1 + 1 * -f();
                _ = self.advance();
                return try self.parseFactor();
            },
            .minus => UnaryOp.neg,
            .exclam => UnaryOp.not,
            else => return try self.parseChain(),
        };
        const val = try self.parseFactor();

        return .{ .unary_op = .{
            .op = op,
            .val = try self.allocNode(val),
        } };
    }

    fn parseChain(self: *@This()) Error!Node {
        // std.log.debug("parse chain", .{});
        var lhs = try self.parseAtom();

        while (true) {
            switch (try self.peekNthToken(0)) {
                .lparen => {
                    // std.log.debug("parse call", .{});
                    const args = try self.parseArgs();
                    const prev = try self.allocNode(lhs); // https://github.com/ziglang/zig/issues/24627
                    lhs = .{ .call = .{
                        .val = prev,
                        .args = args,
                    } };
                },
                .dot => {
                    // std.log.debug("parse field_acc", .{});
                    self.advance();
                    const ident = try self.parseToken(.ident);
                    const prev = try self.allocNode(lhs); // https://github.com/ziglang/zig/issues/24627
                    lhs = .{ .field_acc = .{
                        .val = prev,
                        .ident = ident,
                    } };
                },
                .lbracket => {
                    // std.log.debug("parse index_acc", .{});
                    self.advance();
                    const expr = try self.parseExpr();
                    _ = try self.parseToken(.rbracket);
                    const prev = try self.allocNode(lhs); // https://github.com/ziglang/zig/issues/24627
                    lhs = .{ .index_acc = .{
                        .val = prev,
                        .expr = try self.allocNode(expr),
                    } };
                },
                else => break,
            }
        }

        return lhs;
    }

    fn parseArgs(self: *@This()) Error!NodeRange {
        // std.log.debug("parse args", .{});
        var args: std.ArrayListUnmanaged(Node) = .{};
        defer args.deinit(self.alloc);

        _ = try self.parseToken(.lparen);
        while (true) {
            switch (try self.peekNthToken(0)) {
                .rparen => break,
                else => {},
            }

            try args.append(self.alloc, try self.parseExpr());

            switch (try self.peekNthToken(0)) {
                .rparen => break,
                .comma => self.advance(),
                else => return error.InvalidSyntax,
            }
        }
        _ = try self.parseToken(.rparen);

        return try self.allocNodes(args.items);
    }

    fn parseAtom(self: *@This()) Error!Node {
        // std.log.debug("parse atom", .{});
        switch (try self.peekNthToken(0)) {
            .str_lit => return try self.parseLitStr(),
            .char_lit => return try self.parseLitChar(),
            .float_lit => return try self.parseLitFloat(),
            .int_lit => return try self.parseLitInt(),
            .ident => return .{ .access = .{
                .ident = try self.parseToken(.ident),
            } },
            .lparen => {
                _ = try self.parseToken(.lparen);
                const expr = try self.parseExpr();
                _ = try self.parseToken(.rparen);
                return expr;
            },
            .lbrace => return try self.parseScope(),
            .@"fn" => return try self.parseFn(),
            else => return error.InvalidSyntax,
        }
    }

    fn parseLitStr(self: *@This()) Error!Node {
        // std.log.debug("parse lit str", .{});
        const span = try self.parseToken(.str_lit);
        // .val = self.readSpan(span)[1..][0 .. span.len() - 2],
        return .{ .str_lit = .{
            .tok = span,
        } };
    }

    fn parseLitChar(self: *@This()) Error!Node {
        // std.log.debug("parse lit char", .{});
        const span = try self.parseToken(.char_lit);
        const str = self.readSpan(span);
        // TODO: escapes
        if (str.len != 3) return error.InvalidSyntax;
        return .{ .char_lit = .{
            .tok = span,
            .val = str[1],
        } };
    }

    fn parseLitFloat(self: *@This()) Error!Node {
        // std.log.debug("parse lit float", .{});
        const span = try self.parseToken(.float_lit);

        const base, const str = numberLiteralSplitBase(self.readSpan(span));

        var num: f64 = 0.0;
        var past_dot = false;
        var decimal: f64 = 0.0;
        for (str) |digit_ch| {
            if (digit_ch == '.') {
                past_dot = true;
                continue;
            }

            const digit = std.fmt.charToDigit(digit_ch, base) catch b: {
                std.debug.print("invalid digit in a float literal: '{c}'\n", .{
                    digit_ch,
                });
                break :b 0;
            };

            if (past_dot) {
                decimal -= 1.0;
                num += @as(f64, @floatFromInt(digit)) * std.math.pow(f64, 10.0, decimal);
            } else {
                num *= 10.0;
                num += @floatFromInt(digit);
            }
        }

        return .{ .float_lit = .{
            .val = num,
        } };
    }

    fn parseLitInt(self: *@This()) Error!Node {
        // std.log.debug("parse lit int", .{});
        const span = try self.parseToken(.int_lit);

        const base, const str = numberLiteralSplitBase(self.readSpan(span));

        var num: u64 = 0;
        for (str) |digit_ch| {
            const digit = std.fmt.charToDigit(digit_ch, base) catch b: {
                std.debug.print("invalid digit in an int literal: '{c}'\n", .{
                    digit_ch,
                });
                break :b 0;
            };

            num *= base;
            num += digit;
        }

        return .{ .int_lit = .{
            .val = num,
        } };
    }

    fn numberLiteralSplitBase(str: []const u8) struct { u8, []const u8 } {
        var split_str = str;
        var base: u8 = 10;
        if (str.len >= 2) switch (str[1]) {
            'x' => {
                base = 16;
                split_str = str[2..];
            },
            'o' => {
                base = 8;
                split_str = str[2..];
            },
            'b' => {
                base = 2;
                split_str = str[2..];
            },
            else => {},
        };
        return .{ base, split_str };
    }

    fn parseFn(self: *@This()) !Node {
        // std.log.debug("parse fn", .{});
        _ = try self.parseToken(.@"fn");
        const params = try self.parseParams();
        const scope = try self.parseScope();

        return .{ .@"fn" = .{
            .params = params,
            .scope = try self.allocNode(scope),
        } };
    }

    fn parseParams(self: *@This()) !NodeRange {
        // std.log.debug("parse params", .{});
        var params: std.ArrayListUnmanaged(Node) = .{};
        defer params.deinit(self.alloc);

        _ = try self.parseToken(.lparen);
        while (true) {
            switch (try self.peekNthToken(0)) {
                .rparen => break,
                .comma => {},
                else => return error.InvalidSyntax,
            }

            switch (try self.peekNthToken(0)) {
                .rparen => break,
                .ident => {},
                else => return error.InvalidSyntax,
            }

            const param = try self.parseParam();
            try params.append(self.alloc, param);
        }
        _ = try self.parseToken(.rparen);

        return self.allocNodes(params.items);
    }

    fn parseParam(self: *@This()) !Node {
        // std.log.debug("parse param", .{});
        const ident = try self.parseToken(.ident);
        _ = try self.parseToken(.colon);
        const ty = try self.parseType();

        return .{ .param = .{
            .ident = ident,
            .type = try self.allocNode(ty),
        } };
    }

    fn parseScope(self: *@This()) !Node {
        // std.log.debug("parse scope", .{});
        var stmts: std.ArrayListUnmanaged(Node) = .{};
        defer stmts.deinit(self.alloc);

        var has_trailing_semi = true;

        _ = try self.parseToken(.lbrace);
        while (true) {
            switch (try self.peekNthToken(0)) {
                .rbrace => {
                    self.advance();
                    break;
                },
                .semi => {
                    self.advance();
                    continue;
                },
                else => {},
            }

            const stmt = try self.parseStmt();
            try stmts.append(self.alloc, stmt);

            switch (try self.peekNthToken(0)) {
                .rbrace => {
                    has_trailing_semi = false;
                    self.advance();
                    break;
                },
                else => {},
            }
        }

        return .{ .scope = .{
            .stmts = try self.allocNodes(stmts.items),
            .has_trailing_semi = has_trailing_semi,
        } };
    }

    fn parseStmt(self: *@This()) !Node {
        // std.log.debug("parse stmt", .{});
        const next = try self.peekNthToken(0);
        return switch (next) {
            .let => try self.parseDecl(),
            else => try self.parseExpr(),
        };
    }
};

pub const TypeId = u32;
pub const FieldId = u32;
pub const TypeRange = Range(TypeId, 0);
pub const FieldRange = Range(FieldId, 0);

pub const TypeInfo = union(enum) {
    // comptime_int: u128,
    // comptime_float: f64,
    // usize: void,
    // u128: void,
    u64: void,
    // u32: void,
    // u16: void,
    u8: void,
    // isize: void,
    // i128: void,
    i64: void,
    // i32: void,
    // i16: void,
    i8: void,
    f64: void,
    f32: void,
    bool: void,
    void: void,

    slice: SliceInfo,
    array: ArrayInfo,
    pointer: PointerInfo,
    @"struct": StructInfo,
    @"enum": EnumInfo,
    @"fn": FnInfo,

    pub fn eq(self: @This(), other: @This()) bool {
        // assumes that child types are correctly resolved,
        // so their ids match if they are equal
        return std.meta.eql(self, other);
    }

    pub fn hash(self: @This()) u32 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(@intFromEnum(self));

        switch (self) {
            .comptime_int => |v| hasher.update(std.mem.asBytes(&v)),
            .slice => |v| {
                hasher.update(std.mem.asBytes(&v.elements));
                hasher.update(std.mem.asBytes(&v.mut));
            },
            .array => |v| {
                hasher.update(std.mem.asBytes(&v.elements));
                hasher.update(std.mem.asBytes(&v.len));
            },
            .pointer => |v| {
                hasher.update(std.mem.asBytes(&v.pointee));
                hasher.update(std.mem.asBytes(&v.mut));
            },
            .@"struct" => |v| {
                hasher.update(std.mem.asBytes(&v.fields.start));
                hasher.update(std.mem.asBytes(&v.fields.end));
            },
            .@"enum" => |v| {
                hasher.update(std.mem.asBytes(&v.variants.start));
                hasher.update(std.mem.asBytes(&v.variants.end));
                hasher.update(std.mem.asBytes(&v.tag));
            },
            .@"fn" => |v| {
                hasher.update(std.mem.asBytes(&v.paramns.start));
                hasher.update(std.mem.asBytes(&v.paramns.end));
                hasher.update(std.mem.asBytes(&v.return_ty));
            },
        }

        return @as(u32, @truncate(hasher.final()));
    }
};

pub const SliceInfo = struct {
    elements: TypeId,
    mut: bool,
};

pub const ArrayInfo = struct {
    elements: TypeId,
    len: usize,
};

pub const PointerInfo = struct {
    pointee: TypeId,
    mut: bool,
};

pub const StructInfo = struct {
    fields: FieldRange,
};

pub const EnumInfo = struct {
    variants: FieldRange,
    tag: TypeId,
};

pub const FnInfo = struct {
    paramns: FieldRange,
    return_ty: TypeId,
    @"extern": ?[]const u8,
    ast: NodeId,
};

pub const FieldInfo = struct {
    name: []const u8,
    ty: TypeId,
};

pub const SemanticAnalyzer = struct {
    alloc: std.mem.Allocator,
    types: std.ArrayList(TypeInfo) = .{},
    fields: std.ArrayList(FieldInfo) = .{},
    symbols: Symbols = .{},
    ast_types: []TypeId = &.{},

    ast: []const Node,
    source: []const u8,

    pub const Error = error{
        IncompatibleOperands,
        NotAFunction,
        NotAStruct,
        UnknownField,
        UnknownSymbol,
    } || Symbols.Error;

    pub fn init(
        alloc: std.mem.Allocator,
        ast: []const Node,
        source: []const u8,
    ) @This() {
        return .{
            .alloc = alloc,
            .ast = ast,
            .source = source,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.alloc.free(self.ast_types);
        self.symbols.deinit(self.alloc);
        self.types.deinit(self.alloc);
    }

    pub fn run(
        self: *@This(),
    ) Error!void {
        self.ast_types = try self.alloc.alloc(TypeId, self.ast.len);
        try self.convertStructContents(0);
        self.print();
    }

    pub fn print(self: *@This()) void {
        for (self.types.items, 0..) |type_info, type_id| {
            std.debug.print("type_id {}: {}\n", .{ type_id, type_info });
        }

        self.printNode(0, 0);
    }

    fn indent(depth: usize) void {
        for (0..depth) |_| std.debug.print("| ", .{});
    }

    fn printNode(self: *@This(), node: NodeId, depth: usize) void {
        indent(depth);
        // std.debug.print("id={} ", .{node});

        switch (self.ast[node]) {
            .@"struct" => |v| {
                std.debug.print("struct type_id={}:\n", .{self.ast_types[node]});
                self.printNode(v.contents, depth + 1);
            },
            .struct_contents => |v| {
                std.debug.print("struct_contents type_id={}:\n", .{self.ast_types[node]});
                for (v.fields.start..v.fields.end) |i|
                    self.printNode(@truncate(i), depth + 1);
                for (v.decls.start..v.decls.end) |i|
                    self.printNode(@truncate(i), depth + 1);
            },
            .field => |v| {
                std.debug.print("field name='{s}':\n", .{
                    self.readSpan(v.ident),
                });
                self.printNode(v.type, depth + 1);
                if (v.default) |default|
                    self.printNode(default, depth + 1);
            },
            .decl => |v| {
                std.debug.print("decl mut={} name='{s}':\n", .{
                    v.mut,
                    self.readSpan(v.ident),
                });
                self.printNode(v.expr, depth + 1);
            },
            .@"fn" => |v| {
                const type_id = self.ast_types[node];
                const func = self.types.items[type_id].@"fn";
                std.debug.print("fn type_id={} return_type_id={}:\n", .{
                    type_id,
                    func.return_ty,
                });
                indent(depth + 1);
                std.debug.print("params:\n", .{});
                for (v.params.start..v.params.end) |i|
                    self.printNode(@truncate(i), depth + 2);
                self.printNode(v.scope, depth + 1);
            },
            .scope => |v| {
                std.debug.print("scope type_id={} autoreturn={}:\n", .{
                    self.ast_types[node],
                    !v.has_trailing_semi,
                });
                for (v.stmts.start..v.stmts.end) |i|
                    self.printNode(@truncate(i), depth + 1);
            },
            .param => |v| {
                std.debug.print("param type_id={} name='{s}':\n", .{
                    self.ast_types[v.type],
                    self.readSpan(v.ident),
                });
                self.printNode(v.type, depth + 1);
            },
            .binary_op => |v| {
                std.debug.print("binary_op type_id={} op={t}:\n", .{
                    self.ast_types[node],
                    v.op,
                });
                self.printNode(v.lhs, depth + 1);
                self.printNode(v.rhs, depth + 1);
            },
            .unary_op => |v| {
                std.debug.print("unary_op type_id={} op={t}:\n", .{
                    self.ast_types[node],
                    v.op,
                });
                self.printNode(v.val, depth + 1);
            },
            .field_acc => |v| {
                std.debug.print("field_acc type_id={} name='{s}':\n", .{
                    self.ast_types[node],
                    self.readSpan(v.ident),
                });
                self.printNode(v.val, depth + 1);
            },
            .index_acc => |v| {
                std.debug.print("index_acc type_id={}:\n", .{
                    self.ast_types[node],
                });
                self.printNode(v.val, depth + 1);
                self.printNode(v.expr, depth + 1);
            },
            .call => |v| {
                std.debug.print("call type_id={}:\n", .{
                    self.ast_types[node],
                });
                self.printNode(v.val, depth + 1);
                indent(depth + 1);
                std.debug.print("args:\n", .{});
                for (v.args.start..v.args.end) |i|
                    self.printNode(@truncate(i), depth + 2);
            },
            .access => |v| {
                std.debug.print("access type_id={}: {s}\n", .{
                    self.ast_types[node],
                    self.readSpan(v.ident),
                });
            },
            .str_lit => |v| {
                std.debug.print("str_lit type_id={}: {s}\n", .{
                    self.ast_types[node],
                    self.readSpan(v.tok),
                });
            },
            .char_lit => |v| {
                std.debug.print("char_lit type_id={} raw={}: {s}\n", .{
                    self.ast_types[node],
                    v.val,
                    self.readSpan(v.tok),
                });
            },
            .int_lit => |v| {
                std.debug.print("int_lit type_id={}: {d}\n", .{
                    self.ast_types[node],
                    v.val,
                });
            },
            .float_lit => |v| {
                std.debug.print("float_lit type_id={}: {d}\n", .{
                    self.ast_types[node],
                    v.val,
                });
            },
        }
    }

    fn resolveType(
        self: *@This(),
        ty: TypeInfo,
    ) Error!TypeId {
        // TODO: hashmap for the real compiler
        for (self.types.items, 0..) |existing, type_id| {
            if (existing.eq(ty)) return @intCast(type_id);
        }

        // a new type, add it to the list of types
        const type_id: TypeId = @intCast(self.types.items.len);
        try self.types.append(self.alloc, ty);
        return type_id;
    }

    fn allocFields(
        self: *@This(),
        field_names: []const []const u8,
        field_types: []const TypeId,
    ) Error!FieldRange {
        try self.fields.ensureUnusedCapacity(self.alloc, field_names.len);
        const start: FieldId = @intCast(self.fields.items.len);
        for (field_names, field_types) |name, ty| {
            self.fields.appendAssumeCapacity(.{
                .name = name,
                .ty = ty,
            });
        }

        return .{
            .start = start,
            .end = @intCast(start + field_names.len),
        };
    }

    fn createStruct(
        self: *@This(),
        field_names: []const []const u8,
        field_types: []const TypeId,
    ) Error!TypeId {
        return self.resolveType(.{ .@"struct" = .{
            .fields = try self.allocFields(
                field_names,
                field_types,
            ),
        } });
    }

    fn readSpan(self: *@This(), span: Span) []const u8 {
        return self.source[span.start..span.end];
    }

    fn convertStructContents(
        self: *@This(),
        id: NodeId,
    ) Error!void {
        self.ast_types[id] = try self.resolveType(.void);
        const contents = self.ast[id].struct_contents;
        for (contents.decls.start..contents.decls.end) |i| {
            try self.convertDecl(@intCast(i));
        }
    }

    fn convertStmt(
        self: *SemanticAnalyzer,
        id: NodeId,
    ) Error!?TypeId {
        switch (self.ast[id]) {
            .decl => {
                try self.convertDecl(id);
                return null;
            },
            else => return try self.convertExpr(id),
        }
    }

    fn convertDecl(
        self: *@This(),
        id: NodeId,
    ) Error!void {
        self.ast_types[id] = try self.resolveType(.void);
        const decl = self.ast[id].decl;
        try self.symbols.set(
            self.alloc,
            self.readSpan(decl.ident),
            try self.convertExpr(decl.expr),
        );
    }

    fn convertExpr(
        self: *@This(),
        id: NodeId,
    ) Error!TypeId {
        switch (self.ast[id]) {
            .binary_op => |v| {
                const type_id_lhs = try self.convertExpr(v.lhs);
                const type_id_rhs = try self.convertExpr(v.rhs);
                if (type_id_lhs != type_id_rhs) return Error.IncompatibleOperands;

                switch (v.op) {
                    .add, .sub, .mul, .div, .rem => {
                        self.ast_types[id] = type_id_lhs;
                        return type_id_lhs;
                    },
                }
            },
            .unary_op => |v| {
                const type_id = try self.convertExpr(v.val);

                switch (v.op) {
                    .not, .neg => {
                        self.ast_types[id] = type_id;
                        return type_id;
                    },
                }
            },
            .call => |v| {
                const type_id = try self.convertExpr(v.val);
                const func = switch (self.types.items[type_id]) {
                    .@"fn" => |s| s,
                    else => return Error.NotAFunction,
                };

                self.ast_types[id] = func.return_ty;
                return func.return_ty;
            },
            .field_acc => |v| {
                const type_id_container = try self.convertExpr(v.val);
                const container = switch (self.types.items[type_id_container]) {
                    .@"struct" => |s| s,
                    else => return Error.NotAStruct,
                };

                for (container.fields.start..container.fields.end) |field_id| {
                    const field = self.fields.items[field_id];
                    if (std.mem.eql(u8, field.name, self.readSpan(v.ident))) {
                        const type_id: TypeId = @intCast(field.ty);
                        self.ast_types[id] = type_id;
                        return type_id;
                    }
                }

                return Error.UnknownField;
            },
            .index_acc => |v| {
                _ = v;
                unreachable; // TODO:
            },
            .str_lit => |v| {
                const char = try self.resolveType(.u8);
                const type_id = try self.resolveType(.{ .array = .{
                    .elements = char,
                    .len = v.tok.len() - 2,
                } });
                self.ast_types[id] = type_id;
                return type_id;
            },
            .char_lit => {
                const type_id = try self.resolveType(.u8);
                self.ast_types[id] = type_id;
                return type_id;
            },
            .float_lit => |v| {
                const lossy_cast: f32 = @floatCast(v.val);
                const f32_possible = std.math.approxEqAbs(
                    f64,
                    lossy_cast,
                    v.val,
                    std.math.floatEps(f64),
                );

                const type_id = if (f32_possible)
                    try self.resolveType(.f32)
                else
                    try self.resolveType(.f64);
                self.ast_types[id] = type_id;
                return type_id;
            },
            .int_lit => |v| {
                const Int = struct {
                    T: type,
                    info: TypeInfo,
                };
                inline for ([_]Int{
                    .{ .T = i8, .info = .i8 },
                    .{ .T = i64, .info = .i64 },
                    .{ .T = u8, .info = .u8 },
                    .{ .T = u64, .info = .u64 },
                }) |i| {
                    const lossy_cast = std.math.lossyCast(i.T, v.val);
                    if (lossy_cast == v.val) {
                        const type_id = try self.resolveType(i.info);
                        self.ast_types[id] = type_id;
                        return type_id;
                    }
                }
                unreachable;
            },
            .access => |v| {
                const type_id = self.symbols.get(self.readSpan(v.ident)) orelse
                    return Error.UnknownSymbol;
                self.ast_types[id] = type_id;
                return type_id;
            },
            .scope => return try self.convertScope(id),
            .@"fn" => return try self.convertFn(id),
            else => unreachable,
        }
    }

    fn convertScope(
        self: *SemanticAnalyzer,
        id: NodeId,
    ) Error!TypeId {
        try self.symbols.push(self.alloc);
        defer self.symbols.pop(self.alloc);

        const scope = self.ast[id].scope;
        for (scope.stmts.start..scope.stmts.end) |stmt| {
            _ = try self.convertStmt(@intCast(stmt));
        }

        if (scope.has_trailing_semi or scope.stmts.len() == 0) {
            const type_id = try self.resolveType(.void);
            self.ast_types[id] = type_id;
            return type_id;
        }

        const final_expr = scope.stmts.end - 1;
        if (self.ast[final_expr] == .decl) {
            const type_id = try self.resolveType(.void);
            self.ast_types[id] = type_id;
            return type_id;
        }

        const type_id = self.ast_types[final_expr];
        self.ast_types[id] = type_id;
        return type_id;
    }

    fn convertFn(
        self: *SemanticAnalyzer,
        id: NodeId,
    ) Error!TypeId {
        const f = self.ast[id].@"fn";

        try self.symbols.push(self.alloc);
        defer self.symbols.pop(self.alloc);

        for (f.params.start..f.params.end) |i| {
            const param = self.ast[i].param;
            const type_id = try self.convertType(param.type);
            self.ast_types[i] = type_id;
            try self.symbols.set(
                self.alloc,
                self.readSpan(param.ident),
                type_id,
            );
        }

        try self.fields.ensureUnusedCapacity(self.alloc, f.params.len());
        const start: FieldId = @intCast(self.fields.items.len);
        for (f.params.start..f.params.end) |i| {
            self.fields.appendAssumeCapacity(.{
                .name = self.readSpan(self.ast[i].param.ident),
                .ty = self.ast_types[i],
            });
        }
        const end: FieldId = @intCast(self.fields.items.len);
        const params: FieldRange = .{ .start = start, .end = end };

        const type_id_return = try self.convertScope(f.scope);

        const type_id = try self.resolveType(.{ .@"fn" = .{
            .paramns = params,
            .return_ty = type_id_return,
            .@"extern" = null,
            .ast = id,
        } });
        self.ast_types[id] = type_id;
        return type_id;
    }

    fn convertType(
        self: *SemanticAnalyzer,
        id: NodeId,
    ) Error!TypeId {
        _ = self.ast[id];
        unreachable;
    }
};

pub const Symbols = struct {
    scopes: std.ArrayList(std.StringArrayHashMapUnmanaged(TypeId)) = .{},

    pub const Error = error{OutOfMemory};

    pub fn deinit(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) void {
        for (self.scopes.items) |*scope|
            scope.deinit(alloc);
        self.scopes.deinit(alloc);
    }

    fn lazyinit(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) Error!void {
        if (self.scopes.items.len == 0) {
            @branchHint(.cold);

            // global scope, always there
            try self.scopes.append(alloc, .{});
        }
    }

    pub fn set(
        self: *@This(),
        alloc: std.mem.Allocator,
        var_name: []const u8,
        ty: TypeId,
    ) Error!void {
        try self.lazyinit(alloc);

        const scope = &self.scopes.items[self.scopes.items.len - 1];
        try scope.put(alloc, var_name, ty);
    }

    pub fn get(
        self: *@This(),
        var_name: []const u8,
    ) ?TypeId {
        if (self.scopes.items.len == 0) return null;

        var it = std.mem.reverseIterator(self.scopes.items);
        while (it.next()) |scope| {
            if (scope.get(var_name)) |ty| {
                return ty;
            }
        }
        return null;
    }

    pub fn push(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) Error!void {
        try self.lazyinit(alloc);
        try self.scopes.append(alloc, .{});
    }

    pub fn pop(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) void {
        var scope = self.scopes.pop() orelse return;
        scope.deinit(alloc);
    }
};

// pub const InstrId = u32;
// pub const RegId = u32;

// pub const IrOperation = enum {
//     /// create and initialize a variable
//     let,
//     /// create and initialize a mutable variable
//     let_mut,
//     /// reads the address of a field `a.b`
//     get_field,
//     /// calls a function `a()`
//     call,
// };

// pub const IrOperands = struct {
//     target: RegId,
//     lhs: RegId,
//     rhs: RegId,
//     main: NodeId,
// };

// pub const IrGenerator = struct {
//     alloc: std.mem.Allocator,
//     instr: std.ArrayList(IrOperation) = .{},
//     operands: std.ArrayList(IrOperands) = .{},
//     global_scope: InstrId = 0,

//     ast: []const Node,
//     source: []const u8,

//     pub fn init(
//         alloc: std.mem.Allocator,
//         ast: []const Node,
//         source: []const u8,
//     ) @This() {
//         return .{
//             .alloc = alloc,
//             .ast = ast,
//             .source = source,
//         };
//     }

//     pub fn deinit(self: *@This()) void {
//         self.operands.deinit(self.alloc);
//         self.instr.deinit(self.alloc);
//     }

//     pub fn run(self: *@This()) !void {
//         // measure the avg ir instruction count
//         try self.instr.ensureTotalCapacity(self.alloc, 16);
//         try self.operands.ensureTotalCapacity(self.alloc, 16);

//         self.convertStructContents(self.ast[0].struct_contents);
//     }

//     pub fn convertStructContents(
//         self: *@This(),
//         struct_contents: @FieldType(Node, "struct_contents"),
//     ) void {
//         struct_contents.decls;
//         struct_contents.fields;
//         _ = self; // autofix
//         _ = struct_contents; // autofix

//     }

//     // fn
// };

pub const Codegen = struct {
    alloc: std.mem.Allocator,
    globals: std.ArrayList(u8) = .{},

    ast: []const Node,
    source: []const u8,

    pub fn init(
        alloc: std.mem.Allocator,
        ast: []const Node,
        source: []const u8,
    ) @This() {
        return .{
            .alloc = alloc,
            .ast = ast,
            .source = source,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.globals.deinit(self.alloc);
    }

    pub fn run(self: *@This()) !void {
        _ = self; // autofix
    }

    pub fn dump(
        self: *@This(),
        ir_writer: *std.io.Writer,
    ) !void {
        _ = self;
        _ = try ir_writer.write(
            \\declare i32 @puts(i8*) nounwind
            \\@.hello = private unnamed_addr constant [14 x i8] c"Hello, world!\00"
            \\define i32 @main(i32 %argc, i8** %argv) nounwind {
            \\    %1 = getelementptr [14 x i8], [14 x i8]* @.hello, i32 0, i32 0
            \\    call i32 @puts(i8* %1)
            \\    ret i32 0
            \\}
            \\
        );
    }
};
