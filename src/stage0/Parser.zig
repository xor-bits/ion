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
    assign: struct {
        lhs: NodeId,
        // eq: Span,
        rhs: NodeId,
        // semi: Span,
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
    as,
};

pub const UnaryOp = enum {
    neg,
    not,
};

pub const NodeId = u32;

pub const NodeRange = Range(NodeId, 0);

pub const Error = error{
    InvalidSyntax,
    TooManyAstNodes,
    OutOfMemory,
    EndOfFile,
};

nodes: std.ArrayListUnmanaged(Node) = .{},
tokenizer: *Tokenizer,
current: u32 = 0,

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
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

    const root = try self.allocNode(alloc, .{ .struct_contents = undefined }); // reserve idx 0 as the ast root
    std.debug.assert(root == 0);
    self.nodes.items[0] = try self.parseFile(alloc);
}

fn allocNode(
    self: *@This(),
    alloc: std.mem.Allocator,
    node: Node,
) Error!NodeId {
    // std.log.debug("alloc node={any}", .{node});
    const id = std.math.cast(u32, self.nodes.items.len) orelse
        return error.TooManyAstNodes;
    try self.nodes.append(alloc, node);
    return id;
}

fn allocNodes(
    self: *@This(),
    alloc: std.mem.Allocator,
    nodes: []const Node,
) Error!NodeRange {
    // std.log.debug("alloc nodes={any}", .{nodes});
    const start = std.math.cast(u32, self.nodes.items.len) orelse
        return error.TooManyAstNodes;
    const end = std.math.cast(u32, self.nodes.items.len + nodes.len) orelse
        return error.TooManyAstNodes;

    const slot = try self.nodes.addManyAsSlice(alloc, nodes.len);
    @memcpy(slot, nodes);
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
) Error!Node {
    // std.log.debug("parse file", .{});
    return self.parseStructContents(alloc);
}

fn parseStructContents(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse struct contents", .{});
    var fields: std.ArrayList(Node) = .{};
    defer fields.deinit(alloc);
    var decls: std.ArrayList(Node) = .{};
    defer decls.deinit(alloc);

    while (true) {
        const tok = self.peekToken() catch break;
        switch (tok) {
            .ident => try fields.append(alloc, try self.parseField(alloc)),
            .let => try decls.append(alloc, try self.parseDecl(alloc)),
            .rbrace => break,
            else => return error.InvalidSyntax,
        }
        _ = try self.parseToken(.semi);
    }

    return .{ .struct_contents = .{
        .fields = try self.allocNodes(alloc, fields.items),
        .decls = try self.allocNodes(alloc, decls.items),
    } };
}

fn parseField(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse field", .{});
    const ident = try self.parseToken(.ident);
    const colon = try self.parseToken(.colon);
    const ty = try self.parseExpr(alloc);

    const tok = try self.peekToken();
    switch (tok) {
        .eq => {
            const eq = try self.parseToken(.eq);
            const default = try self.parseExpr(alloc);
            const comma = try self.parseToken(.comma);

            _ = .{ colon, eq, comma };

            return .{ .field = .{
                .ident = ident,
                .type = try self.allocNode(alloc, ty),
                .default = try self.allocNode(alloc, default),
            } };
        },
        .comma => {
            const comma = try self.parseToken(.comma);

            _ = .{ colon, comma };

            return .{ .field = .{
                .ident = ident,
                .type = try self.allocNode(alloc, ty),
                .default = null,
            } };
        },
        else => return error.InvalidSyntax,
    }
}

fn parseDecl(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
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
    const eq = try self.parseToken(.eq);
    const expr = try self.parseExpr(alloc);
    // const semi = try self.parseToken(.semi);

    _ = .{ let, eq };

    return .{ .decl = .{
        .mut = mut != null,
        .ident = ident,
        .type_hint = type_hint,
        .expr = try self.allocNode(alloc, expr),
    } };
}

fn parseExpr(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse expr", .{});
    switch (try self.peekToken()) {
        .lbracket => return self.parseSlice(alloc),
        .asterisk => return self.parsePointer(alloc),
        else => return self.parseAssign(alloc),
    }
}

fn parseSlice(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    _ = try self.parseToken(.lbracket);

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
            const elements = try self.allocNode(alloc, try self.parseExpr(alloc));

            return Node{ .slice = .{
                .elements_expr = elements,
                .mut = mut,
            } };
        },
        else => {
            // is an array, compile time length

            const length = try self.allocNode(alloc, try self.parseExpr(alloc));
            _ = try self.parseToken(.rbracket);
            const elements = try self.allocNode(alloc, try self.parseExpr(alloc));

            return Node{ .array = .{
                .length_expr = length,
                .elements_expr = elements,
            } };
        },
    }
}

fn parsePointer(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    _ = try self.parseToken(.asterisk);

    var mut: bool = false;
    switch (try self.peekToken()) {
        .mut => {
            self.advance();
            mut = true;
        },
        else => {},
    }
    const pointee = try self.allocNode(alloc, try self.parseExpr(alloc));

    return Node{ .pointer = .{
        .pointee_expr = pointee,
        .mut = mut,
    } };
}

fn parseAssign(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse assign", .{});
    var lhs = try self.parseCast(alloc);

    if (try self.peekToken() == .eq) {
        _ = self.advance();
        const rhs = try self.parseCast(alloc);

        const prev = try self.allocNode(alloc, lhs); // https://github.com/ziglang/zig/issues/24627
        lhs = .{ .assign = .{
            .lhs = prev,
            .rhs = try self.allocNode(alloc, rhs),
        } };
    }

    return lhs;
}

fn parseCast(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse cast", .{});
    var lhs = try self.parseSum(alloc);

    while (true) {
        const op = switch (try self.peekToken()) {
            .as => BinaryOp.as,
            else => break,
        };
        _ = self.advance();
        const rhs = try self.parseSum(alloc);

        const prev = try self.allocNode(alloc, lhs); // https://github.com/ziglang/zig/issues/24627
        lhs = .{ .binary_op = .{
            .lhs = prev,
            .op = op,
            .rhs = try self.allocNode(alloc, rhs),
        } };
    }

    return lhs;
}

fn parseSum(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse sum", .{});
    var lhs = try self.parseProd(alloc);

    while (true) {
        const op = switch (try self.peekToken()) {
            .asterisk => BinaryOp.mul,
            .slash => BinaryOp.div,
            .percent => BinaryOp.rem,
            else => break,
        };
        _ = self.advance();
        const rhs = try self.parseProd(alloc);

        const prev = try self.allocNode(alloc, lhs); // https://github.com/ziglang/zig/issues/24627
        lhs = .{ .binary_op = .{
            .lhs = prev,
            .op = op,
            .rhs = try self.allocNode(alloc, rhs),
        } };
    }

    return lhs;
}

fn parseProd(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse term", .{});
    var lhs = try self.parseFactor(alloc);

    while (true) {
        const op = switch (try self.peekToken()) {
            .plus => BinaryOp.add,
            .minus => BinaryOp.sub,
            else => break,
        };
        _ = self.advance();
        const rhs = try self.parseFactor(alloc);

        const prev = try self.allocNode(alloc, lhs); // https://github.com/ziglang/zig/issues/24627
        lhs = .{ .binary_op = .{
            .lhs = prev,
            .op = op,
            .rhs = try self.allocNode(alloc, rhs),
        } };
    }

    return lhs;
}

fn parseFactor(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse factor", .{});
    const op = switch (try self.peekToken()) {
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

    return .{ .unary_op = .{
        .op = op,
        .val = try self.allocNode(alloc, val),
    } };
}

fn parseChain(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse chain", .{});
    var lhs = try self.parseAtom(alloc);

    while (true) {
        switch (try self.peekToken()) {
            .lparen => {
                // std.log.debug("parse call", .{});
                const args = try self.parseArgs(alloc);
                const prev = try self.allocNode(alloc, lhs); // https://github.com/ziglang/zig/issues/24627
                lhs = .{ .call = .{
                    .val = prev,
                    .args = args,
                } };
            },
            .dot => {
                // std.log.debug("parse field_acc", .{});
                self.advance();
                const ident = try self.parseToken(.ident);
                const prev = try self.allocNode(alloc, lhs); // https://github.com/ziglang/zig/issues/24627
                lhs = .{ .field_acc = .{
                    .val = prev,
                    .ident = ident,
                } };
            },
            .lbracket => {
                // std.log.debug("parse index_acc", .{});
                self.advance();
                const expr = try self.parseExpr(alloc);
                _ = try self.parseToken(.rbracket);
                const prev = try self.allocNode(alloc, lhs); // https://github.com/ziglang/zig/issues/24627
                lhs = .{ .index_acc = .{
                    .val = prev,
                    .expr = try self.allocNode(alloc, expr),
                } };
            },
            else => break,
        }
    }

    return lhs;
}

fn parseArgs(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!NodeRange {
    // std.log.debug("parse args", .{});
    var args: std.ArrayListUnmanaged(Node) = .{};
    defer args.deinit(alloc);

    _ = try self.parseToken(.lparen);
    while (true) {
        switch (try self.peekToken()) {
            .rparen => break,
            else => {},
        }

        try args.append(alloc, try self.parseExpr(alloc));

        switch (try self.peekToken()) {
            .rparen => break,
            .comma => self.advance(),
            else => return error.InvalidSyntax,
        }
    }
    _ = try self.parseToken(.rparen);

    return try self.allocNodes(alloc, args.items);
}

fn parseAtom(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!Node {
    // std.log.debug("parse atom", .{});
    switch (try self.peekToken()) {
        .str_lit => return try self.parseLitStr(),
        .char_lit => return try self.parseLitChar(),
        .float_lit => return try self.parseLitFloat(),
        .int_lit => return try self.parseLitInt(),
        .ident => return .{ .access = .{
            .ident = try self.parseToken(.ident),
        } },
        .lparen => {
            _ = try self.parseToken(.lparen);
            const expr = try self.parseExpr(alloc);
            _ = try self.parseToken(.rparen);
            return expr;
        },
        .lbrace => return try self.parseScope(alloc),
        .@"fn", .@"extern" => return try self.parseFn(alloc),
        else => return error.InvalidSyntax,
    }
}

fn parseLitStr(
    self: *@This(),
) Error!Node {
    // std.log.debug("parse lit str", .{});
    const span = try self.parseToken(.str_lit);
    // .val = self.readSpan(span)[1..][0 .. span.len() - 2],
    return .{ .str_lit = .{
        .tok = span,
    } };
}

fn parseLitChar(
    self: *@This(),
) Error!Node {
    // std.log.debug("parse lit char", .{});
    const span = try self.parseToken(.char_lit);
    const str = span.read(self.tokenizer.source);
    // TODO: escapes
    if (str.len != 3) return error.InvalidSyntax;
    return .{ .char_lit = .{
        .tok = span,
        .val = str[1],
    } };
}

fn parseLitFloat(
    self: *@This(),
) Error!Node {
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

    return .{ .float_lit = .{
        .val = num,
    } };
}

fn parseLitInt(
    self: *@This(),
) Error!Node {
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

    return .{ .int_lit = .{
        .val = num,
    } };
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
) !Node {
    // std.log.debug("parse fn", .{});
    const proto = try self.parseProto(alloc);

    const next_tok = try self.peekToken();
    if (proto.proto.@"extern" and next_tok == .at) {
        self.advance();
        const symexpr = try self.parseExpr(alloc);

        return .{ .@"fn" = .{
            .proto = try self.allocNode(alloc, proto),
            .scope_or_symexpr = try self.allocNode(alloc, symexpr),
        } };
    } else if (next_tok == .lbrace) {
        const scope = try self.parseScope(alloc);

        return .{ .@"fn" = .{
            .proto = try self.allocNode(alloc, proto),
            .scope_or_symexpr = try self.allocNode(alloc, scope),
        } };
    } else {
        return proto;
    }
}

fn parseProto(
    self: *@This(),
    alloc: std.mem.Allocator,
) !Node {
    // std.log.debug("parse proto", .{});
    var is_extern = false;
    switch (try self.peekToken()) {
        .@"extern" => {
            is_extern = true;
            self.advance();
        },
        else => {},
    }

    _ = try self.parseToken(.@"fn");
    const params = try self.parseParams(alloc);

    var return_ty_expr: ?NodeId = null;
    switch (try self.peekToken()) {
        .colon => {
            self.advance();
            return_ty_expr = try self.allocNode(alloc, try self.parseExpr(alloc));
        },
        else => {},
    }

    return .{ .proto = .{
        .@"extern" = is_extern,
        .params = params.params,
        .is_va_args = params.is_va_args,
        .return_ty_expr = return_ty_expr,
    } };
}

fn parseParams(
    self: *@This(),
    alloc: std.mem.Allocator,
) !struct {
    params: NodeRange,
    is_va_args: bool,
} {
    // std.log.debug("parse params", .{});
    var params: std.ArrayListUnmanaged(Node) = .{};
    defer params.deinit(alloc);

    var va_args = false;

    _ = try self.parseToken(.lparen);
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
        try params.append(alloc, param);

        switch (try self.peekToken()) {
            .rparen => break,
            .comma => self.advance(),
            else => {},
        }
    }
    _ = try self.parseToken(.rparen);

    return .{
        .params = try self.allocNodes(alloc, params.items),
        .is_va_args = va_args,
    };
}

fn parseParam(
    self: *@This(),
    alloc: std.mem.Allocator,
) !Node {
    // std.log.debug("parse param", .{});
    const ident = try self.parseToken(.ident);
    _ = try self.parseToken(.colon);
    const ty = try self.allocNode(alloc, try self.parseExpr(alloc));

    return .{ .param = .{
        .ident = ident,
        .type = ty,
    } };
}

fn parseScope(
    self: *@This(),
    alloc: std.mem.Allocator,
) !Node {
    // std.log.debug("parse scope", .{});
    var stmts: std.ArrayListUnmanaged(Node) = .{};
    defer stmts.deinit(alloc);

    var has_trailing_semi = true;

    _ = try self.parseToken(.lbrace);
    while (true) {
        switch (try self.peekToken()) {
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

        const stmt = try self.parseStmt(alloc);
        try stmts.append(alloc, stmt);

        switch (try self.peekToken()) {
            .rbrace => {
                has_trailing_semi = false;
                self.advance();
                break;
            },
            else => {},
        }
    }

    return .{ .scope = .{
        .stmts = try self.allocNodes(alloc, stmts.items),
        .has_trailing_semi = has_trailing_semi,
    } };
}

fn parseStmt(
    self: *@This(),
    alloc: std.mem.Allocator,
) !Node {
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
