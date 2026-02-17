const std = @import("std");
const Range = @import("main.zig").Range;
const Tokenizer = @import("Tokenizer.zig");
const Token = Tokenizer.Token;
const Span = Tokenizer.Span;
const SpannedToken = Tokenizer.SpannedToken;

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
        type_hint: ?NodeId,
        // eq: Span,
        expr: NodeId,
        // semi: Span,
    },
    @"if": struct {
        // @"if": Span,
        check_expr: NodeId,
        on_true_scope: NodeId,
        on_false_scope: NodeId,
    },
    // @"for": struct {
    //     // @"for": Span,
    //     check_expr: NodeId,
    //     on_true_scope: NodeId,
    //     on_false_scope: NodeId,
    // },
    loop: struct {
        // @"loop": Span,
        scope: NodeId,
    },
    assign: struct {
        lhs: NodeId,
        // eq: Span,
        rhs: NodeId,
        // semi: Span,
    },
    proto: struct {
        @"extern": bool,
        // @"fn": Span,
        // lparen: Span,
        params: NodeRange,
        is_va_args: bool,
        // rparen: Span,
        // colon: Span,
        return_ty_expr: ?NodeId,
    },
    @"fn": struct {
        proto: NodeId,
        scope_or_symexpr: NodeId,
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
    array: struct {
        // lbracket: Span,
        length_expr: NodeId,
        // rbracket: Span,
        elements_expr: NodeId,
    },
    // TODO: pratt parsing + make these (slice, pointer, etc.) into unary ops
    slice: struct {
        // lbracket: Span,
        // rbracket: Span,
        mut: bool,
        elements_expr: NodeId,
    },
    pointer: struct {
        // asterisk: Span,
        mut: bool,
        pointee_expr: NodeId,
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

pub const SpannedNode = struct {
    node: Node,
    span: Span,
};

pub const BinaryOp = enum {
    add,
    sub,
    mul,
    div,
    rem,
    as,

    eq,
    neq,
    lt,
    le,
    gt,
    ge,

    field,

    pub fn format(self: *const @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
        return writer.print("{s}", .{switch (self.*) {
            .add => "+",
            .sub => "-",
            .mul => "*",
            .div => "/",
            .rem => "%",
            .as => "as",

            .eq => "==",
            .neq => "!=",
            .lt => "<",
            .le => "<=",
            .gt => ">",
            .ge => ">=",

            .field => ".",
        }});
    }
};

pub const UnaryOp = enum {
    neg,
    not,
    slice,
    slice_mut,
    pointer,
    pointer_mut,

    pub fn format(self: *const @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
        return writer.print("{s}", .{switch (self.*) {
            .neg => "-",
            .not => "!",
            .slice => "[]",
            .slice_mut => "[]mut",
            .pointer => "*",
            .pointer_mut => "*mut",
        }});
    }
};

pub const NodeId = u32;

pub const NodeRange = Range(NodeId, 0);

pub const Error = error{
    InvalidSyntax,
    TooManyAstNodes,
    OutOfMemory,
    EndOfFile,
};

nodes: std.ArrayList(Node) = .{},
node_spans: std.ArrayList(Span) = .{},
// TODO: node_stack: std.ArrayList(Node) = .{},
tokenizer: *Tokenizer,
current: u32 = 0,

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    self.node_spans.deinit(alloc);
    self.nodes.deinit(alloc);
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) !void {
    errdefer {
        if (self.peek()) |current| {
            std.debug.print("parse error at token: {s}\n", .{
                current.span.read(self.tokenizer.source),
            });
        } else |_| {}
    }

    // assume approx one node per 3 characters
    try self.nodes.ensureTotalCapacity(alloc, self.tokenizer.source.len / 3);
    self.nodes.clearRetainingCapacity();

    const root = try self.allocNode(alloc, undefined); // reserve idx 0 as the ast root
    std.debug.assert(root == 0);
    const root_file = try self.parseFile(alloc);
    self.nodes.items[0] = root_file.node;
    self.node_spans.items[0] = root_file.span;
}

fn allocNode(
    self: *@This(),
    alloc: std.mem.Allocator,
    node: SpannedNode,
) Error!NodeId {
    // std.log.debug("alloc node={any}", .{node});
    const id = std.math.cast(u32, self.nodes.items.len) orelse
        return error.TooManyAstNodes;
    try self.nodes.append(alloc, node.node);
    try self.node_spans.append(alloc, node.span);
    return id;
}

fn allocNodes(
    self: *@This(),
    alloc: std.mem.Allocator,
    nodes: []const Node,
    spans: []const Span,
) Error!NodeRange {
    std.debug.assert(nodes.len == spans.len);
    const n = nodes.len;

    // std.log.debug("alloc nodes={any}", .{nodes});
    const start = std.math.cast(u32, self.nodes.items.len) orelse
        return error.TooManyAstNodes;
    const end = std.math.cast(u32, self.nodes.items.len + n) orelse
        return error.TooManyAstNodes;

    const nodes_dst = try self.nodes.addManyAsSlice(alloc, n);
    const spans_dst = try self.node_spans.addManyAsSlice(alloc, n);
    @memcpy(nodes_dst, nodes);
    @memcpy(spans_dst, spans);
    return .{ .start = start, .end = end };
}

fn peek(
    self: *@This(),
) Error!SpannedToken {
    if (self.current >= self.tokenizer.tokens.len)
        return Error.EndOfFile;
    return self.tokenizer.tokens.get(self.current);
}

fn peekNth(
    self: *@This(),
    n: usize,
) Error!SpannedToken {
    if (self.current + n >= self.tokenizer.tokens.len)
        return Error.EndOfFile;
    return self.tokenizer.tokens.get(self.current + n);
}

fn peekToken(
    self: *@This(),
) Error!Token {
    return (try self.peek()).token;
}

fn peekSpan(
    self: *@This(),
) Error!Span {
    return (try self.peek()).span;
}

fn peekTokenNth(
    self: *@This(),
    n: usize,
) Error!Token {
    return (try self.peekNth(n)).token;
}

fn peekSpanNth(
    self: *@This(),
    n: usize,
) Error!Span {
    return (try self.peekNth(n)).span;
}

fn popIfEql(
    self: *@This(),
    expect: Token,
) ?SpannedToken {
    const tok = self.peek() catch return null;
    if (tok.token != expect) return null;

    self.advance();
    return tok;
}

fn advance(
    self: *@This(),
) void {
    self.current += 1;
}

fn parseToken(
    self: *@This(),
    expect: Token,
) error{InvalidSyntax}!Span {
    const tok = self.popIfEql(expect) orelse
        return error.InvalidSyntax;
    return tok.span;
}

fn parseFile(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse file", .{});
    return self.parseStructContents(alloc);
}

fn parseStructContents(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse struct contents", .{});
    // TODO: use a shared stack instead
    var field_nodes: std.ArrayList(Node) = .{};
    defer field_nodes.deinit(alloc);
    var field_spans: std.ArrayList(Span) = .{};
    defer field_spans.deinit(alloc);
    var decl_nodes: std.ArrayList(Node) = .{};
    defer decl_nodes.deinit(alloc);
    var decl_spans: std.ArrayList(Span) = .{};
    defer decl_spans.deinit(alloc);

    var span: Span = Span{};

    while (true) {
        const tok = self.peekToken() catch break;
        switch (tok) {
            .ident => {
                const field = try self.parseField(alloc);
                try field_nodes.append(alloc, field.node);
                try field_spans.append(alloc, field.span);
                span = span.merge(field.span);
            },
            .let => {
                const decl = try self.parseDecl(alloc);
                try decl_nodes.append(alloc, decl.node);
                try decl_spans.append(alloc, decl.span);
                span = span.merge(decl.span);
            },
            .rbrace => break,
            else => return error.InvalidSyntax,
        }
        span = span.merge(try self.parseToken(.semi));
    }

    return .{
        .node = .{ .struct_contents = .{
            .fields = try self.allocNodes(
                alloc,
                field_nodes.items,
                field_spans.items,
            ),
            .decls = try self.allocNodes(
                alloc,
                decl_nodes.items,
                decl_spans.items,
            ),
        } },
        .span = span,
    };
}

fn parseField(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse field", .{});
    const ident = try self.parseToken(.ident);
    const colon = try self.parseToken(.colon);
    const ty = try self.parseExpr(alloc);

    const tok = try self.peekToken();
    switch (tok) {
        .single_eq => {
            const eq = try self.parseToken(.single_eq);
            const default = try self.parseExpr(alloc);
            const comma = try self.parseToken(.comma);

            _ = .{ colon, eq, comma };

            return .{
                .node = .{ .field = .{
                    .ident = ident,
                    .type = try self.allocNode(alloc, ty),
                    .default = try self.allocNode(alloc, default),
                } },
                .span = ident.merge(comma),
            };
        },
        .comma => {
            const comma = try self.parseToken(.comma);

            _ = .{ colon, comma };

            return .{
                .node = .{ .field = .{
                    .ident = ident,
                    .type = try self.allocNode(alloc, ty),
                    .default = null,
                } },
                .span = ident.merge(comma),
            };
        },
        else => return error.InvalidSyntax,
    }
}

fn parseDecl(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse decl", .{});
    const let = try self.parseToken(.let);
    var mut: ?Span = null;
    if (try self.peekToken() == .mut)
        mut = try self.parseToken(.mut);

    const ident = try self.parseToken(.ident);
    var type_hint: ?NodeId = null;
    if (try self.peekToken() == .colon) {
        self.advance();
        type_hint = try self.allocNode(alloc, try self.parseExpr(alloc));
    }
    const eq = try self.parseToken(.single_eq);
    const expr = try self.parseExpr(alloc);
    // const semi = try self.parseToken(.semi);

    _ = .{ let, eq };

    return .{
        .node = .{ .decl = .{
            .mut = mut != null,
            .ident = ident,
            .type_hint = type_hint,
            .expr = try self.allocNode(alloc, expr),
        } },
        .span = let.merge(expr.span),
    };
}

fn parseExpr(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse expr", .{});
    return self.parseComparison(alloc);
}

fn parseSlice(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    const lbracket = try self.parseToken(.lbracket);

    switch (try self.peekToken()) {
        .rbracket => {
            self.advance();
            // is a slice, runtime length

            var mut: bool = false;
            switch (try self.peekToken()) {
                .mut => {
                    self.advance();
                    mut = true;
                },
                else => {},
            }
            const elements = try self.parseExpr(alloc);

            return .{
                .node = .{ .slice = .{
                    .elements_expr = try self.allocNode(alloc, elements),
                    .mut = mut,
                } },
                .span = lbracket.merge(elements.span),
            };
        },
        else => {
            // is an array, compile time length

            const length = try self.parseExpr(alloc);
            _ = try self.parseToken(.rbracket);
            const elements = try self.parseExpr(alloc);

            return .{
                .node = .{ .array = .{
                    .length_expr = try self.allocNode(alloc, length),
                    .elements_expr = try self.allocNode(alloc, elements),
                } },
                .span = lbracket.merge(elements.span),
            };
        },
    }
}

fn parsePointer(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    const asterisk = try self.parseToken(.asterisk);

    var mut: bool = false;
    switch (try self.peekToken()) {
        .mut => {
            self.advance();
            mut = true;
        },
        else => {},
    }
    const pointee = try self.parseExpr(alloc);

    return .{
        .node = .{ .pointer = .{
            .pointee_expr = try self.allocNode(alloc, pointee),
            .mut = mut,
        } },
        .span = asterisk.merge(pointee.span),
    };
}

fn parseComparison(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse comparison", .{});
    var lhs: SpannedNode = try self.parseAssign(alloc);

    while (true) {
        const op = switch (self.peekToken() catch break) {
            .double_eq => BinaryOp.eq,
            .neq => BinaryOp.neq,
            .lt => BinaryOp.lt,
            .le => BinaryOp.le,
            .gt => BinaryOp.gt,
            .ge => BinaryOp.ge,
            else => break,
        };
        _ = self.advance();
        const rhs = try self.parseAssign(alloc);

        // FIXME: this is a workaround to a bug in the Zig compiler
        // https://github.com/ziglang/zig/issues/24627
        const lhs_copy = lhs;
        lhs = .{
            .node = .{ .binary_op = .{
                .lhs = try self.allocNode(alloc, lhs_copy),
                .op = op,
                .rhs = try self.allocNode(alloc, rhs),
            } },
            .span = lhs_copy.span.merge(rhs.span),
        };
    }

    return lhs;
}

fn parseAssign(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse assign", .{});
    var lhs: SpannedNode = try self.parseCast(alloc);

    if (try self.peekToken() == .single_eq) {
        _ = self.advance();
        const rhs = try self.parseCast(alloc);

        // FIXME: this is a workaround to a bug in the Zig compiler
        // https://github.com/ziglang/zig/issues/24627
        const lhs_copy = lhs;
        lhs = .{
            .node = .{ .assign = .{
                .lhs = try self.allocNode(alloc, lhs_copy),
                .rhs = try self.allocNode(alloc, rhs),
            } },
            .span = lhs_copy.span.merge(rhs.span),
        };
    }

    return lhs;
}

fn parseCast(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse cast", .{});
    var lhs: SpannedNode = try self.parseSum(alloc);

    while (true) {
        const op = switch (try self.peekToken()) {
            .as => BinaryOp.as,
            else => break,
        };
        _ = self.advance();
        const rhs = try self.parseSum(alloc);

        // FIXME: this is a workaround to a bug in the Zig compiler
        // https://github.com/ziglang/zig/issues/24627
        const lhs_copy = lhs;
        lhs = .{
            .node = .{ .binary_op = .{
                .lhs = try self.allocNode(alloc, lhs_copy),
                .op = op,
                .rhs = try self.allocNode(alloc, rhs),
            } },
            .span = lhs_copy.span.merge(rhs.span),
        };
    }

    return lhs;
}

fn parseSum(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse sum", .{});
    var lhs: SpannedNode = try self.parseProd(alloc);

    while (true) {
        const op = switch (try self.peekToken()) {
            .asterisk => BinaryOp.mul,
            .slash => BinaryOp.div,
            .percent => BinaryOp.rem,
            else => break,
        };
        _ = self.advance();
        const rhs = try self.parseProd(alloc);

        // FIXME: this is a workaround to a bug in the Zig compiler
        // https://github.com/ziglang/zig/issues/24627
        const lhs_copy = lhs;
        lhs = .{
            .node = .{ .binary_op = .{
                .lhs = try self.allocNode(alloc, lhs_copy),
                .op = op,
                .rhs = try self.allocNode(alloc, rhs),
            } },
            .span = lhs_copy.span.merge(rhs.span),
        };
    }

    return lhs;
}

fn parseProd(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse term", .{});
    var lhs: SpannedNode = try self.parseFactor(alloc);

    while (true) {
        const op = switch (try self.peekToken()) {
            .plus => BinaryOp.add,
            .minus => BinaryOp.sub,
            else => break,
        };
        _ = self.advance();
        const rhs = try self.parseFactor(alloc);

        // FIXME: this is a workaround to a bug in the Zig compiler
        // https://github.com/ziglang/zig/issues/24627
        const lhs_copy = lhs;
        lhs = .{
            .node = .{ .binary_op = .{
                .lhs = try self.allocNode(alloc, lhs_copy),
                .op = op,
                .rhs = try self.allocNode(alloc, rhs),
            } },
            .span = lhs_copy.span.merge(rhs.span),
        };
    }

    return lhs;
}

fn parseFactor(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse factor", .{});
    const op_tok = try self.peek();
    const op = switch (op_tok.token) {
        .plus => {
            // 1 + 1 * -f();
            _ = self.advance();
            return try self.parseFactor(alloc);
        },
        .minus => UnaryOp.neg,
        .exclam => UnaryOp.not,
        else => return try self.parseChain(alloc),
    };
    const val = try self.parseFactor(alloc);

    return .{
        .node = .{ .unary_op = .{
            .op = op,
            .val = try self.allocNode(alloc, val),
        } },
        .span = op_tok.span.merge(val.span),
    };
}

fn parseChain(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse chain", .{});
    var lhs: SpannedNode = try self.parseAtom(alloc);

    while (true) {
        switch (try self.peekToken()) {
            .lparen => {
                // std.log.debug("parse call", .{});
                const args, const args_span = try self.parseArgs(alloc);
                // FIXME: this is a workaround to a bug in the Zig compiler
                // https://github.com/ziglang/zig/issues/24627
                const lhs_copy = lhs;
                lhs = .{
                    .node = .{ .call = .{
                        .val = try self.allocNode(alloc, lhs_copy),
                        .args = args,
                    } },
                    .span = lhs_copy.span.merge(args_span),
                };
            },
            .single_dot => {
                // std.log.debug("parse field_acc", .{});
                self.advance();
                const ident = try self.parseToken(.ident);
                // FIXME: this is a workaround to a bug in the Zig compiler
                // https://github.com/ziglang/zig/issues/24627
                const lhs_copy = lhs;
                lhs = .{
                    .node = .{ .field_acc = .{
                        .val = try self.allocNode(alloc, lhs_copy),
                        .ident = ident,
                    } },
                    .span = lhs_copy.span.merge(ident),
                };
            },
            .lbracket => {
                // std.log.debug("parse index_acc", .{});
                self.advance();
                const expr = try self.parseExpr(alloc);
                _ = try self.parseToken(.rbracket);
                // FIXME: this is a workaround to a bug in the Zig compiler
                // https://github.com/ziglang/zig/issues/24627
                const lhs_copy = lhs;
                lhs = .{
                    .node = .{ .index_acc = .{
                        .val = try self.allocNode(alloc, lhs_copy),
                        .expr = try self.allocNode(alloc, expr),
                    } },
                    .span = lhs_copy.span.merge(expr.span),
                };
            },
            else => break,
        }
    }

    return lhs;
}

fn parseArgs(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!struct { NodeRange, Span } {
    // std.log.debug("parse args", .{});
    var arg_nodes: std.ArrayListUnmanaged(Node) = .{};
    defer arg_nodes.deinit(alloc);
    var arg_spans: std.ArrayListUnmanaged(Span) = .{};
    defer arg_spans.deinit(alloc);

    const lparen = try self.parseToken(.lparen);
    while (true) {
        switch (try self.peekToken()) {
            .rparen => break,
            else => {},
        }
        const expr = try self.parseExpr(alloc);

        try arg_nodes.append(alloc, expr.node);
        try arg_spans.append(alloc, expr.span);

        switch (try self.peekToken()) {
            .rparen => break,
            .comma => self.advance(),
            else => return error.InvalidSyntax,
        }
    }
    const rparen = try self.parseToken(.rparen);

    return .{
        try self.allocNodes(
            alloc,
            arg_nodes.items,
            arg_spans.items,
        ),
        lparen.merge(rparen),
    };
}

fn parseAtom(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!SpannedNode {
    // std.log.debug("parse atom", .{});
    switch (try self.peekToken()) {
        .str_lit => return try self.parseLitStr(),
        .char_lit => return try self.parseLitChar(),
        .float_lit => return try self.parseLitFloat(),
        .int_lit => return try self.parseLitInt(),
        .ident => {
            const ident = try self.parseToken(.ident);
            return .{
                .node = .{ .access = .{
                    .ident = ident,
                } },
                .span = ident,
            };
        },
        .lparen => {
            const lparen = try self.parseToken(.lparen);
            const expr = try self.parseExpr(alloc);
            const rparen = try self.parseToken(.rparen);
            return .{
                .node = expr.node,
                .span = lparen.merge(rparen),
            };
        },
        .lbrace => return try self.parseScope(alloc),
        .@"fn", .@"extern" => return try self.parseFn(alloc),
        .@"if" => return try self.parseIf(alloc),
        .loop => return try self.parseLoop(alloc),
        .lbracket => return self.parseSlice(alloc),
        .asterisk => return self.parsePointer(alloc),
        else => return error.InvalidSyntax,
    }
}

fn parseLitStr(
    self: *@This(),
) Error!SpannedNode {
    // std.log.debug("parse lit str", .{});
    const span = try self.parseToken(.str_lit);
    // .val = self.readSpan(span)[1..][0 .. span.len() - 2],
    return .{
        .node = .{ .str_lit = .{
            .tok = span,
        } },
        .span = span,
    };
}

fn parseLitChar(
    self: *@This(),
) Error!SpannedNode {
    // std.log.debug("parse lit char", .{});
    const span = try self.parseToken(.char_lit);
    const str = span.read(self.tokenizer.source);
    // TODO: escapes
    if (str.len != 3) return error.InvalidSyntax;
    return .{
        .node = .{ .char_lit = .{
            .tok = span,
            .val = str[1],
        } },
        .span = span,
    };
}

fn parseLitFloat(
    self: *@This(),
) Error!SpannedNode {
    // std.log.debug("parse lit float", .{});
    const span = try self.parseToken(.float_lit);

    const base, const str = numberLiteralSplitBase(span.read(self.tokenizer.source));

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

    return .{
        .node = .{ .float_lit = .{
            .val = num,
        } },
        .span = span,
    };
}

fn parseLitInt(
    self: *@This(),
) Error!SpannedNode {
    // std.log.debug("parse lit int", .{});
    const span = try self.parseToken(.int_lit);

    const base, const str = numberLiteralSplitBase(span.read(self.tokenizer.source));

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

    return .{
        .node = .{ .int_lit = .{
            .val = num,
        } },
        .span = span,
    };
}

fn numberLiteralSplitBase(
    str: []const u8,
) struct { u8, []const u8 } {
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

fn parseFn(
    self: *@This(),
    alloc: std.mem.Allocator,
) !SpannedNode {
    // std.log.debug("parse fn", .{});
    const proto = try self.parseProto(alloc);

    const next_tok = try self.peekToken();
    if (proto.node.proto.@"extern" and next_tok == .at) {
        self.advance();
        const symexpr = try self.parseExpr(alloc);

        return .{
            .node = .{ .@"fn" = .{
                .proto = try self.allocNode(alloc, proto),
                .scope_or_symexpr = try self.allocNode(alloc, symexpr),
            } },
            .span = proto.span.merge(symexpr.span),
        };
    } else if (next_tok == .lbrace) {
        const scope = try self.parseScope(alloc);

        return .{
            .node = .{ .@"fn" = .{
                .proto = try self.allocNode(alloc, proto),
                .scope_or_symexpr = try self.allocNode(alloc, scope),
            } },
            .span = proto.span.merge(scope.span),
        };
    } else {
        return proto;
    }
}

fn parseProto(
    self: *@This(),
    alloc: std.mem.Allocator,
) !SpannedNode {
    // std.log.debug("parse proto", .{});
    var is_extern = false;
    const maybe_extern_tok = try self.peek();
    switch (maybe_extern_tok.token) {
        .@"extern" => {
            is_extern = true;
            self.advance();
        },
        else => {},
    }

    _ = try self.parseToken(.@"fn");
    const params = try self.parseParams(alloc);

    var return_ty_expr: ?NodeId = null;
    var end_span = params.span;
    switch (try self.peekToken()) {
        .colon => {
            self.advance();
            const expr = try self.parseExpr(alloc);
            return_ty_expr = try self.allocNode(alloc, expr);
            end_span = expr.span;
        },
        else => {},
    }

    return .{
        .node = .{ .proto = .{
            .@"extern" = is_extern,
            .params = params.params,
            .is_va_args = params.is_va_args,
            .return_ty_expr = return_ty_expr,
        } },
        .span = maybe_extern_tok.span.merge(end_span),
    };
}

fn parseParams(
    self: *@This(),
    alloc: std.mem.Allocator,
) !struct {
    params: NodeRange,
    span: Span,
    is_va_args: bool,
} {
    // std.log.debug("parse params", .{});
    var param_nodes: std.ArrayListUnmanaged(Node) = .{};
    defer param_nodes.deinit(alloc);
    var param_spans: std.ArrayListUnmanaged(Span) = .{};
    defer param_spans.deinit(alloc);

    var va_args = false;

    const lparen = try self.parseToken(.lparen);
    while (true) {
        switch (try self.peekToken()) {
            .rparen => break,
            .double_dot => {
                va_args = true;
                self.advance();
                break;
            },
            else => {},
        }

        const param = try self.parseParam(alloc);
        try param_nodes.append(alloc, param.node);
        try param_spans.append(alloc, param.span);

        switch (try self.peekToken()) {
            .rparen => break,
            .comma => self.advance(),
            else => {},
        }
    }
    const rparen = try self.parseToken(.rparen);

    return .{
        .params = try self.allocNodes(
            alloc,
            param_nodes.items,
            param_spans.items,
        ),
        .span = lparen.merge(rparen),
        .is_va_args = va_args,
    };
}

fn parseParam(
    self: *@This(),
    alloc: std.mem.Allocator,
) !SpannedNode {
    // std.log.debug("parse param", .{});
    const ident = try self.parseToken(.ident);
    _ = try self.parseToken(.colon);
    const ty_expr = try self.parseExpr(alloc);
    const ty = try self.allocNode(alloc, ty_expr);

    return .{
        .node = .{ .param = .{
            .ident = ident,
            .type = ty,
        } },
        .span = ident.merge(ty_expr.span),
    };
}

fn parseIf(
    self: *@This(),
    alloc: std.mem.Allocator,
) !SpannedNode {
    const if_kw = try self.parseToken(.@"if");
    const check_expr = try self.parseExpr(alloc);
    const on_true_scope = try self.parseScope(alloc);
    var on_false_scope: SpannedNode = undefined;

    if (self.peekToken() catch .invalid_byte == .@"else") {
        _ = try self.parseToken(.@"else");

        if (self.peekToken() catch .invalid_byte == .@"if") {
            // TODO: does not need to reCurse
            const nested_if = try self.parseIf(alloc);
            const nested_if_id = try self.allocNode(alloc, nested_if);
            on_false_scope = .{
                .node = .{ .scope = .{
                    .stmts = .{ .start = nested_if_id, .end = nested_if_id + 1 },
                    .has_trailing_semi = true,
                } },
                .span = nested_if.span,
            };
        } else {
            on_false_scope = try self.parseScope(alloc);
        }
    } else {
        on_false_scope = .{
            .node = .{ .scope = .{
                .stmts = .{},
                .has_trailing_semi = true,
            } },
            .span = on_true_scope.span,
        };
    }

    return .{
        .node = .{ .@"if" = .{
            .check_expr = try self.allocNode(alloc, check_expr),
            .on_true_scope = try self.allocNode(alloc, on_true_scope),
            .on_false_scope = try self.allocNode(alloc, on_false_scope),
        } },
        .span = if_kw.merge(on_false_scope.span),
    };
}

fn parseLoop(
    self: *@This(),
    alloc: std.mem.Allocator,
) !SpannedNode {
    const loop = try self.parseToken(.loop);

    const scope = try self.parseScope(alloc);
    return .{
        .node = .{ .loop = .{
            .scope = try self.allocNode(alloc, scope),
        } },
        .span = loop.merge(scope.span),
    };
}

fn parseScope(
    self: *@This(),
    alloc: std.mem.Allocator,
) !SpannedNode {
    // std.log.debug("parse scope", .{});
    var stmt_nodes: std.ArrayListUnmanaged(Node) = .{};
    defer stmt_nodes.deinit(alloc);
    var stmt_spans: std.ArrayListUnmanaged(Span) = .{};
    defer stmt_spans.deinit(alloc);

    var has_trailing_semi = true;

    const lbrace = try self.parseToken(.lbrace);
    while (true) {
        switch (try self.peekToken()) {
            .rbrace => break,
            .semi => {
                self.advance();
                continue;
            },
            else => {},
        }

        const stmt = try self.parseStmt(alloc);
        try stmt_nodes.append(alloc, stmt.node);
        try stmt_spans.append(alloc, stmt.span);

        switch (try self.peekToken()) {
            .rbrace => {
                has_trailing_semi = false;
                break;
            },
            else => {},
        }
    }
    const rbrace = try self.parseToken(.rbrace);

    return .{
        .node = .{ .scope = .{
            .stmts = try self.allocNodes(
                alloc,
                stmt_nodes.items,
                stmt_spans.items,
            ),
            .has_trailing_semi = has_trailing_semi,
        } },
        .span = lbrace.merge(rbrace),
    };
}

fn parseStmt(
    self: *@This(),
    alloc: std.mem.Allocator,
) !SpannedNode {
    // std.log.debug("parse stmt", .{});
    const next = try self.peekToken();
    return switch (next) {
        .let => try self.parseDecl(alloc),
        else => try self.parseExpr(alloc),
    };
}

pub fn dump(
    self: *const @This(),
) void {
    std.debug.print("PARSER DUMP:\n", .{});
    for (self.nodes.items, 0..) |node, i|
        std.debug.print("id={} node={}\n", .{ i, node });
    self.print(0, 0);
}

fn indent(
    depth: usize,
) void {
    for (0..depth) |_| std.debug.print("| ", .{});
}

fn print(
    self: *const @This(),
    node: NodeId,
    depth: usize,
) void {
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
                v.ident.read(self.tokenizer.source),
            });
            self.print(v.type, depth + 1);
            if (v.default) |default|
                self.print(default, depth + 1);
        },
        .decl => |v| {
            std.debug.print("decl mut={} name='{s}':\n", .{
                v.mut,
                v.ident.read(self.tokenizer.source),
            });
            self.print(v.expr, depth + 1);
        },
        .@"if" => |v| {
            std.debug.print("if:\n", .{});
            indent(depth + 1);
            std.debug.print("check:\n", .{});
            self.print(v.check_expr, depth + 2);
            indent(depth + 1);
            std.debug.print("on_true:\n", .{});
            self.print(v.on_true_scope, depth + 2);
            indent(depth + 1);
            std.debug.print("on_false:\n", .{});
            self.print(v.on_false_scope, depth + 2);
        },
        .loop => |v| {
            std.debug.print("loop scope:\n", .{});
            self.print(v.scope, depth + 1);
        },
        .assign => |v| {
            std.debug.print("assign:\n", .{});
            self.print(v.lhs, depth + 1);
            self.print(v.rhs, depth + 1);
        },
        .@"fn" => |v| {
            std.debug.print("fn:\n", .{});
            indent(depth + 1);
            std.debug.print("proto:\n", .{});
            self.print(v.proto, depth + 2);
            indent(depth + 1);
            if (self.nodes.items[v.proto].proto.@"extern")
                std.debug.print("symexpr:\n", .{})
            else
                std.debug.print("scope:\n", .{});
            self.print(v.scope_or_symexpr, depth + 2);
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
                v.ident.read(self.tokenizer.source),
            });
            self.print(v.type, depth + 1);
        },
        .array => |v| {
            std.debug.print("array:\n", .{});
            indent(depth + 1);
            std.debug.print("length:\n", .{});
            self.print(v.length_expr, depth + 2);
            indent(depth + 1);
            std.debug.print("element:\n", .{});
            self.print(v.elements_expr, depth + 2);
        },
        .slice => |v| {
            std.debug.print("slice mut={}:\n", .{v.mut});
            self.print(v.elements_expr, depth + 1);
        },
        .pointer => |v| {
            std.debug.print("pointer mut={}:\n", .{v.mut});
            self.print(v.pointee_expr, depth + 1);
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
                v.ident.read(self.tokenizer.source),
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
                v.ident.read(self.tokenizer.source),
            });
        },
        .proto => |v| {
            std.debug.print("proto extern={}:\n", .{
                v.@"extern",
            });
            if (v.return_ty_expr) |return_ty_expr| {
                indent(depth + 1);
                std.debug.print("return:\n", .{});
                self.print(return_ty_expr, depth + 2);
            }
            indent(depth + 1);
            std.debug.print("params:\n", .{});
            for (v.params.start..v.params.end) |i|
                self.print(@truncate(i), depth + 2);
        },
        .str_lit => |v| {
            std.debug.print("str_lit: {s}\n", .{
                v.tok.read(self.tokenizer.source),
            });
        },
        .char_lit => |v| {
            std.debug.print("char_lit raw={}: {s}\n", .{
                v.val,
                v.tok.read(self.tokenizer.source),
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
