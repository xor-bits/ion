const std = @import("std");

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

    var buffer: [0x8000]u8 = undefined;

    var source_reader = source_file.reader(&buffer);
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

    return 0;
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

pub const Span = struct {
    start: u32 = 0,
    end: u32 = 0,

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
                        '.' => .dot,
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

pub const NodeRange = struct {
    low: NodeId,
    high: NodeId,
};

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
                for (v.fields.low..v.fields.high) |i|
                    self.print(@truncate(i), depth + 1);
                for (v.decls.low..v.decls.high) |i|
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
                for (v.params.low..v.params.high) |i|
                    self.print(@truncate(i), depth + 2);
                self.print(v.scope, depth + 1);
            },
            .scope => |v| {
                std.debug.print("scope autoreturn={}:\n", .{
                    !v.has_trailing_semi,
                });
                for (v.stmts.low..v.stmts.high) |i|
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
                for (v.args.low..v.args.high) |i|
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
        return .{ .low = start, .high = end };
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
