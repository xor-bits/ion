const std = @import("std");
const Range = @import("main.zig").Range;

pub const Token = enum {
    let,
    mut,
    @"fn",
    @"extern",
    as,
    @"if",
    @"else",
    @"for",
    loop,
    @"return",
    @"break",
    @"continue",
    print,

    semi,
    colon,
    comma,
    single_dot,
    double_dot,
    lparen,
    rparen,
    lbrace,
    rbrace,
    lbracket,
    rbracket,

    single_eq,
    double_eq,
    neq,
    lt,
    le,
    gt,
    ge,

    ampersand,
    at,
    exclam,
    plus,
    minus,
    asterisk,
    slash,
    percent,

    ident,
    str_lit,
    char_lit,
    int_lit,
    float_lit,

    invalid_token,
    eof,

    pub fn isSimple(
        self: @This(),
    ) bool {
        return switch (self) {
            .ident,
            .str_lit,
            .char_lit,
            .int_lit,
            .float_lit,
            .invalid_token,
            .eof,
            => false,
            else => true,
        };
    }

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        const str: []const u8 = switch (self) {
            .semi => ";",
            .colon => ":",
            .comma => ",",
            .single_dot => ".",
            .double_dot => "..",
            .lparen => "(",
            .rparen => ")",
            .lbrace => "{",
            .rbrace => "}",
            .lbracket => "[",
            .rbracket => "]",

            .single_eq => "=",
            .double_eq => "==",
            .neq => "!=",
            .lt => "<",
            .le => "<=",
            .gt => ">",
            .ge => ">=",

            .ampersand => "&",
            .at => "@",
            .exclam => "!",
            .plus => "+",
            .minus => "-",
            .asterisk => "*",
            .slash => "/",
            .percent => "%",

            else => @tagName(self),
        };
        const simple = self.isSimple();
        if (simple) try writer.writeByte('\'');
        try writer.writeAll(str);
        if (simple) try writer.writeByte('\'');
    }
};

const Keyword = enum {
    let,
    mut,
    @"fn",
    @"extern",
    as,
    @"if",
    @"else",
    @"for",
    loop,
    @"return",
    @"break",
    @"continue",
    print,
};

pub const Span = Range(u32, 0);

pub const SpannedToken = struct {
    token: Token,
    span: Span,
};

eof: SpannedToken = .{
    .token = .eof,
    .span = .{},
},
tokens: std.MultiArrayList(SpannedToken) = .{},
cursor: u32 = 0,

source: []const u8,

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    self.tokens.deinit(alloc);
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) !void {
    try self.tokens.ensureUnusedCapacity(alloc, self.source.len / 2);

    while (self.next()) |tok| {
        try self.tokens.append(alloc, tok);
    }

    if (self.tokens.len != 0) {
        const last_real_token = self.tokens.get(self.tokens.len - 1);
        self.eof.span = .{
            .start = last_real_token.span.end,
            .end = last_real_token.span.end,
        };
    }
}

pub fn dump(
    self: *@This(),
) void {
    for (0..self.tokens.len) |i| {
        const tok = self.tokens.get(i);
        std.debug.print("{t:>10} : `{s}`\n", .{
            tok.token,
            self.readSpan(tok.span),
        });
    }
}

fn next(
    self: *@This(),
) ?SpannedToken {
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
            '_',
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
                        ch, span = self.popIfEql('.') orelse break :b .single_dot;
                        break :b .double_dot;
                    },
                    '(' => .lparen,
                    ')' => .rparen,
                    '{' => .lbrace,
                    '}' => .rbrace,
                    '[' => .lbracket,
                    ']' => .rbracket,

                    '=' => b: {
                        ch, span = self.popIfEql('=') orelse break :b .single_eq;
                        break :b .double_eq;
                    },
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
                    '&' => .ampersand,
                    '@' => .at,
                    '+' => .plus,
                    '-' => .minus,
                    '*' => .asterisk,
                    '/' => .slash,
                    '%' => .percent,

                    else => return .{
                        .token = .invalid_token,
                        .span = start_span.merge(span),
                    },
                };
                const full_span = start_span.merge(span);

                return .{
                    .token = simple_tok,
                    .span = full_span,
                };
            },
        },
        .comment => switch (ch) {
            '/' => continue :loop State.line_comment,
            '*' => continue :loop State.block_comment,
            else => return .{
                // just a `/` token
                .token = Token.slash,
                .span = start_span.merge(span),
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
                ch, span = self.popIf(isIdentChar) orelse break;
            }

            const full_span = start_span.merge(span);
            const ident = self.readSpan(full_span);

            const keyword = std.meta.stringToEnum(Keyword, ident) orelse {
                return .{
                    .token = Token.ident,
                    .span = full_span,
                };
            };

            switch (keyword) {
                inline else => |kw| {
                    return .{
                        .token = @field(Token, @tagName(kw)),
                        .span = full_span,
                    };
                },
            }
        },
        .str_lit => {
            while (ch != '"') {
                ch, span = self.pop() orelse break;
            }

            return .{
                .token = Token.str_lit,
                .span = start_span.merge(span),
            };
        },
        .char_lit => {
            _ = self.pop() orelse return null;
            ch, span = self.pop() orelse return null;

            return .{
                .token = Token.char_lit,
                .span = start_span.merge(span),
            };
        },
        .num_lit => {
            var dot_found = false;
            while (true) {
                const result = self.peek() orelse break;

                if (dot_found and result.@"0" == '.') break;
                if (result.@"0" == '.') {
                    // the character after a dot has to
                    // be a digit for the dot to be accpected
                    //
                    // this is to parse 3..4 as a <3> <..> <4> instead of <3.> . <4>
                    const digit_after_dot, _ = self.peekNth(1) orelse break;
                    if (!std.ascii.isDigit(digit_after_dot)) break;
                    dot_found = true;
                } else if (!std.ascii.isDigit(result.@"0")) break;
                ch, span = result;
                self.advance();
            }

            const full_span = start_span.merge(span);

            return if (dot_found) .{
                .token = Token.float_lit,
                .span = full_span,
            } else .{
                .token = Token.int_lit,
                .span = full_span,
            };
        },
    }
}

fn popIf(
    self: *@This(),
    comptime pred: fn (u8) bool,
) ?struct { u8, Span } {
    const ch, const span = self.peek() orelse return null;
    if (pred(ch)) {
        self.advance();
        return .{ ch, span };
    } else {
        return null;
    }
}

fn popIfEql(
    self: *@This(),
    expect: u8,
) ?struct { u8, Span } {
    const ch, const span = self.peek() orelse return null;
    if (ch == expect) {
        self.advance();
        return .{ ch, span };
    } else {
        return null;
    }
}

fn pop(
    self: *@This(),
) ?struct { u8, Span } {
    defer self.advance();
    return self.peek();
}

fn peek(
    self: *@This(),
) ?struct { u8, Span } {
    return self.peekNth(0);
}

fn peekNth(
    self: *@This(),
    n: u32,
) ?struct { u8, Span } {
    if (self.cursor + n >= self.source.len) return null;
    return .{
        self.source[self.cursor + n],
        Span{ .start = self.cursor + n, .end = self.cursor + n + 1 },
    };
}

fn advance(
    self: *@This(),
) void {
    if (self.cursor >= self.source.len) return;
    self.cursor += 1;
}

fn readSpan(
    self: *@This(),
    span: Span,
) []const u8 {
    return self.source[span.start..span.end];
}

fn isIdentChar(
    c: u8,
) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}
