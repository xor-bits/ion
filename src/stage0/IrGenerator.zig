const std = @import("std");
const Parser = @import("Parser.zig");
const Tokenizer = @import("Tokenizer.zig");
const Span = Tokenizer.Span;
const Node = Parser.Node;
const NodeId = Parser.NodeId;
const Range = @import("main.zig").Range;

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

pub const InstrId = struct { u32 };
pub const InstrRange = Range(InstrId, .{0});
pub const RegId = packed struct {
    idx: u30 = 0,
    kind: enum(u2) {
        local,
        global,
        func,
    } = .local,

    pub fn next(
        counter: *@This(),
    ) @This() {
        defer counter.idx += 1;
        return counter.*;
    }

    pub fn format(
        self: *const @This(),
        writer: *std.io.Writer,
    ) std.io.Writer.Error!void {
        const prefix = switch (self.kind) {
            .local => "%",
            .global => "g",
            .func => "f",
        };
        try writer.print("{s}{}", .{
            prefix,
            self.idx,
        });
    }
};

pub const Instr = union(enum) {
    /// create and initialize a variable
    let: struct {
        ident: Span,
        mut: bool,
        value: RegId,
    },
    /// create and initialize a meta variable: parameter
    let_param: struct {
        ident: Span,
        mut: bool,
        value: RegId,
    },
    /// reads the address of a field `a.b`
    get_field: struct {
        result: RegId,
        container: RegId,
        field_idx: u32,
    },

    str_lit: struct {
        result: RegId,
        value: Span,
    },
    int_lit: struct {
        result: RegId,
        value: u128,
    },
    float_lit: struct {
        result: RegId,
        value: f64,
    },
    void_lit: struct {
        result: RegId,
    },

    // /// always followed by another `.struct_field` or `.struct_finish`
    // struct_field: struct {
    //     field_ty: RegId,
    // },
    // /// creates a new struct type with fields given by earlier `.struct_field`s
    // struct_finish: struct {},

    // /// always followed by another `.func_param` or `.func_finish`
    // func_param: struct {
    //     param_ty: RegId,
    // },
    // /// creates a new function
    // func_finish: struct {},

    /// always followed by another `.call_arg`, `.call_finish` or '.call_finish_builtin'
    call_arg: struct {
        value: RegId,
    },
    /// calls a function `a()` with arguments given by earlier `.call_arg`s
    call_finish: struct {
        result: RegId,
        /// function to be called
        func: RegId,
    },
    /// calls a function `a()` with arguments given by earlier `.call_arg`s
    call_finish_builtin: struct {
        result: RegId,
        /// function to be called
        func: BuiltinFunc,
    },

    pub fn allocatesRegister(self: @This()) bool {
        switch (self) {}
    }
};

pub const BuiltinFunc = enum {
    add,
    sub,
    mul,
    div,
    rem,

    neg,
    not,
};

pub const Block = struct {
    instructions: InstrRange,
    branch: union(enum) {
        @"return": RegId,
    },
};

pub const Static = struct {};

pub const Prototype = struct {
    /// meta block is used to evaluate parameter and return types
    meta: Block,
};

pub const Function = struct {
    symbol: []const u8,
    /// meta block is used to evaluate parameter and return types
    meta: Block,
    /// main block is the actual code, before monomorphizing
    main: Block,
};

pub const Namespace = struct {};

pub const Global = struct {
    symbol: []const u8,
    block: Block,
};

pub const Error = error{
    OutOfMemory,
    UnknownSymbol,
};

instr: std.MultiArrayList(Instr) = .{},
// statics: std.MultiArrayList(Static) = .{},
functions: std.MultiArrayList(Function) = .{},
// prototypes: std.MultiArrayList(Prototype) = .{},
// namespaces: std.MultiArrayList(Namespace) = .{},
globals: std.ArrayList(Global) = .{},
global: RegId = .{ .kind = .global },
local: RegId = .{ .kind = .local },
symbols: Symbols = .{},
builder: Builder = .{},

// root_namespace: InstrId = 0,
parser: *Parser,

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    self.builder.deinit(alloc);
    self.symbols.deinit(alloc);
    for (self.globals.items) |global| {
        alloc.free(global.symbol);
    }
    self.globals.deinit(alloc);
    for (0..self.functions.len) |i| {
        alloc.free(self.functions.get(i).symbol);
    }
    self.functions.deinit(alloc);
    self.instr.deinit(alloc);
}

fn nodes(
    self: *@This(),
) []const Node {
    return self.parser.nodes.items;
}

fn source(
    self: *@This(),
) []const u8 {
    return self.parser.tokenizer.source;
}

fn pushFunction(
    self: *@This(),
) RegId {
    defer self.local = .{ .kind = .local };
    return self.local;
}

fn popFunction(
    self: *@This(),
    function_ptr: RegId,
) void {
    self.local = function_ptr;
    self.local.idx += 1;
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!void {
    // measure the avg ir instruction count
    try self.instr.ensureTotalCapacity(alloc, 16);

    try self.symbols.push(alloc);
    defer self.symbols.pop(alloc);

    try self.convertStructContents(
        alloc,
        &.{ .part = "<root>" },
        self.nodes()[0].struct_contents,
    );
}

pub fn dump(
    self: *@This(),
) void {
    std.debug.print("IRGEN DUMP:\n", .{});
    for (self.globals.items, 0..) |global, idx| {
        std.debug.print(
            \\{f} = global {s}:
            \\  setup:
            \\
        , .{
            RegId{ .idx = @intCast(idx), .kind = .global },
            global.symbol,
        });
        self.dumpBlock(global.block);
    }
    for (0..self.functions.len) |idx| {
        const function = self.functions.get(idx);
        std.debug.print(
            \\{f} = function {s}:
            \\  setup:
            \\
        , .{
            RegId{ .idx = @intCast(idx), .kind = .func },
            function.symbol,
        });
        self.dumpBlock(function.meta);
        std.debug.print(
            \\  entry:
            \\
        , .{});
        self.dumpBlock(function.main);
        std.debug.print(
            \\
        , .{});
    }
}

fn dumpBlock(
    self: *@This(),
    block: Block,
) void {
    for (block.instructions.start.@"0"..block.instructions.end.@"0") |instr_i| {
        self.dumpInstr(self.instr.get(instr_i));
    }
    std.debug.print("    @return({f})\n", .{
        block.branch.@"return",
    });
}

fn dumpInstr(
    self: *@This(),
    instr: Instr,
) void {
    switch (instr) {
        .let => |v| {
            _ = v;
            @panic("todo");
        },
        .let_param => |v| {
            _ = v;
            @panic("todo");
        },
        .get_field => |v| {
            std.debug.print("    {f} = @get_field({f}, {})\n", .{
                v.result,
                v.container,
                v.field_idx,
            });
        },

        .str_lit => |v| {
            std.debug.print("    {f} = {s}\n", .{
                v.result,
                v.value.read(self.parser.tokenizer.source),
            });
        },
        .int_lit => |v| {
            std.debug.print("    {f} = {}\n", .{
                v.result,
                v.value,
            });
        },
        .float_lit => |v| {
            std.debug.print("    {f} = {}\n", .{
                v.result,
                v.value,
            });
        },
        .void_lit => |v| {
            std.debug.print("    {f} = {{}}\n", .{
                v.result,
            });
        },

        .call_arg => |v| {
            std.debug.print("    @call_arg({f})\n", .{
                v.value,
            });
        },
        .call_finish => |v| {
            std.debug.print("    {f} = @call_finish({f})\n", .{
                v.result,
                v.func,
            });
        },
        .call_finish_builtin => |v| {
            std.debug.print("    {f} = @call_finish_builtin({s})\n", .{
                v.result,
                @tagName(v.func),
            });
        },
    }
}

pub fn convertStructContents(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    struct_contents: @FieldType(Node, "struct_contents"),
) Error!void {
    for (struct_contents.decls.start..struct_contents.decls.end) |i| {
        try self.convertGlobalDecl(
            alloc,
            name_hint,
            @intCast(i),
        );
    }
}

pub fn convertGlobalDecl(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const decl = self.nodes()[node_id].decl;
    const next_name_hint = name_hint.push(decl.ident.read(self.source()));

    if (decl.type_hint) |type_hint| {
        _ = type_hint;
        @panic("todo");
        // const type_hint = try self.convertExpr(alloc, &.{
        //     .prev = name_hint,
        //     .part = decl.ident.read(self.parser.tokenizer.source),
        // }, type_hint);
    }

    try self.builder.pushBlock(alloc);

    const name = decl.ident.read(self.parser.tokenizer.source);
    const local_val = try self.convertExpr(
        alloc,
        &next_name_hint,
        decl.expr,
    );

    const block = try self.builder.popBlock(alloc, &self.instr);
    const global_val: RegId = .{ .idx = @intCast(self.globals.items.len), .kind = .global };
    try self.globals.append(alloc, .{
        .symbol = try next_name_hint.generate(alloc),
        .block = .{
            .instructions = block,
            .branch = .{ .@"return" = local_val },
        },
    });

    try self.symbols.set(alloc, name, global_val);
}

pub fn convertLocalDecl(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const decl = self.nodes()[node_id].decl;
    const next_name_hint = name_hint.push(decl.ident.read(self.source()));

    if (decl.type_hint) |type_hint| {
        _ = type_hint;
        @panic("todo");
        // const type_hint = try self.convertExpr(alloc, &.{
        //     .prev = name_hint,
        //     .part = decl.ident.read(self.parser.tokenizer.source),
        // }, type_hint);
    }

    const name = decl.ident.read(self.parser.tokenizer.source);
    const val = try self.convertExpr(
        alloc,
        &next_name_hint,
        decl.expr,
    );

    try self.symbols.set(alloc, name, val);
}

pub fn convertFn(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    const func = self.nodes()[node_id].@"fn";
    const proto = self.nodes()[func.proto].proto;
    if (proto.@"extern") @panic("todo");

    try self.symbols.push(alloc);
    defer self.symbols.pop(alloc);

    const fn_ptr_val = self.pushFunction();
    defer self.popFunction(fn_ptr_val);

    const prototype = try self.convertProtoBare(alloc, name_hint, func.proto);
    const function = try self.convertFnBare(alloc, name_hint, node_id, prototype);

    const func_idx = self.functions.len;
    try self.functions.append(alloc, function);

    return RegId{
        .idx = @intCast(func_idx),
        .kind = .func,
    };
}

fn convertFnBare(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
    prototype: Prototype,
) Error!Function {
    const func = self.nodes()[node_id].@"fn";

    try self.builder.pushBlock(alloc);

    const scope = try self.convertScope(alloc, name_hint, func.scope_or_symexpr);

    // TODO: scope should build the block
    const main_block = try self.builder.popBlock(alloc, &self.instr);
    return .{
        .symbol = try name_hint.generate(alloc),
        .meta = prototype.meta,
        .main = .{
            .instructions = main_block,
            .branch = .{ .@"return" = scope },
        },
    };
}

pub fn convertProto(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Prototype {
    const proto = self.nodes()[node_id].proto;

    try self.symbols.push(alloc);
    defer self.symbols.pop(alloc);

    const fn_ptr_val = self.pushFunction();
    defer self.popFunction(fn_ptr_val);

    return try self.convertProtoBare(alloc, name_hint, proto);
}

fn convertProtoBare(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Prototype {
    const proto = self.nodes()[node_id].proto;

    try self.builder.pushBlock(alloc);

    for (proto.params.start..proto.params.end) |i| {
        const param = self.nodes()[i].param;
        const param_name = param.ident.read(self.parser.tokenizer.source);
        const param_type = try self.convertExpr(alloc, name_hint, param.type);

        try self.symbols.set(alloc, param_name, param_type);
    }

    const return_ty = if (proto.return_ty_expr) |i| b: {
        break :b try self.convertExpr(alloc, name_hint, i);
    } else b: {
        const result = self.local.next();
        try self.builder.pushOne(alloc, .{ .void_lit = .{
            .result = result,
        } });
        break :b result;
    };

    const meta_block = try self.builder.popBlock(alloc, &self.instr);
    return .{
        .meta = .{
            .instructions = meta_block,
            .branch = .{ .@"return" = return_ty },
        },
    };
}

pub fn convertExpr(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    switch (self.nodes()[node_id]) {
        .array => @panic("todo"),
        .slice => @panic("todo"),
        .pointer => @panic("todo"),
        .binary_op => |v| {
            const lhs = try self.convertExpr(alloc, name_hint, v.lhs);
            const rhs = try self.convertExpr(alloc, name_hint, v.rhs);

            try self.builder.pushPrepare(alloc, 3);
            const result = self.local.next();
            self.builder.push(.{ .call_arg = .{
                .value = lhs,
            } });
            self.builder.push(.{ .call_arg = .{
                .value = rhs,
            } });
            self.builder.push(.{ .call_finish_builtin = .{
                .result = result,
                .func = switch (v.op) {
                    inline else => |op| @field(BuiltinFunc, @tagName(op)),
                },
            } });

            return result;
        },
        .unary_op => |v| {
            const val = try self.convertExpr(alloc, name_hint, v.val);

            try self.builder.pushPrepare(alloc, 2);
            const result = self.local.next();
            self.builder.push(.{ .call_arg = .{
                .value = val,
            } });
            self.builder.push(.{ .call_finish_builtin = .{
                .result = result,
                .func = switch (v.op) {
                    inline else => |op| @field(BuiltinFunc, @tagName(op)),
                },
            } });

            return result;
        },
        .call => |v| {
            const func = try self.convertExpr(alloc, name_hint, v.val);

            const arg_regs = try alloc.alloc(RegId, v.args.len());
            defer alloc.free(arg_regs);

            for (arg_regs, v.args.start..v.args.end) |*arg, i| {
                arg.* = try self.convertExpr(alloc, name_hint, @intCast(i));
            }

            const result = self.local.next();
            try self.builder.pushPrepare(alloc, @intCast(v.args.len() + 1));
            for (arg_regs) |arg| {
                self.builder.push(.{ .call_arg = .{
                    .value = arg,
                } });
            }
            self.builder.push(.{ .call_finish = .{
                .result = result,
                .func = func,
            } });

            return result;
        },
        .field_acc => {
            // const container = try self.convertExpr(alloc, name_hint, v.val);

            // self.instr.append(alloc, .{ .get_field = .{
            //     .container = container,
            //     // .field_idx = ,
            // } });

            @panic("todo");
        },
        .index_acc => {
            @panic("todo");
        },
        .str_lit => |v| {
            var value = v.tok;
            value.start += 1;
            value.end -= 1;

            const result = self.local.next();
            try self.builder.pushOne(alloc, .{ .str_lit = .{
                .result = result,
                .value = value,
            } });
            return result;
        },
        .char_lit => |v| {
            const result = self.local.next();
            try self.builder.pushOne(alloc, .{ .int_lit = .{
                .result = result,
                .value = v.val,
            } });
            return result;
        },
        .float_lit => |v| {
            const result = self.local.next();
            try self.builder.pushOne(alloc, .{ .float_lit = .{
                .result = result,
                .value = v.val,
            } });
            return result;
        },
        .int_lit => |v| {
            const result = self.local.next();
            try self.builder.pushOne(alloc, .{ .int_lit = .{
                .result = result,
                .value = v.val,
            } });
            return result;
        },
        .access => |v| {
            const sym = v.ident.read(self.parser.tokenizer.source);
            return self.symbols.get(sym) orelse {
                std.debug.print("unknown symbol: {s}\n", .{sym});
                return error.UnknownSymbol;
            };
        },
        .scope => return try self.convertScope(
            alloc,
            &name_hint.push("<scope>"),
            node_id,
        ),
        .@"fn" => return try self.convertFn(
            alloc,
            name_hint,
            node_id,
        ),
        else => std.debug.panic("TODO: {}", .{self.nodes()[node_id]}),
    }
}

pub fn convertScope(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    try self.symbols.push(alloc);
    defer self.symbols.pop(alloc);

    const scope = self.nodes()[node_id].scope;

    const stmts, const result_expr = scope.stmts.splitLast() orelse {
        const result = self.local.next();
        try self.builder.pushOne(alloc, .{ .void_lit = .{
            .result = result,
        } });
        return result;
    };

    for (stmts.start..stmts.end) |i| {
        try self.convertStmt(alloc, name_hint, @intCast(i));
    }

    if (scope.has_trailing_semi) {
        return try self.convertExpr(alloc, name_hint, result_expr);
    } else {
        try self.convertStmt(alloc, name_hint, result_expr);

        const result = self.local.next();
        try self.builder.pushOne(alloc, .{ .void_lit = .{
            .result = result,
        } });
        return result;
    }
}

pub fn convertStmt(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    switch (self.nodes()[node_id]) {
        .decl => {
            try self.convertLocalDecl(
                alloc,
                name_hint,
                node_id,
            );
        },
        else => {
            _ = try self.convertExpr(
                alloc,
                name_hint,
                node_id,
            );
        },
    }
}

pub const Symbols = struct {
    scopes: std.ArrayList(std.StringArrayHashMapUnmanaged(RegId)) = .{},

    pub fn deinit(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) void {
        for (self.scopes.items) |*scope|
            scope.deinit(alloc);
        self.scopes.deinit(alloc);
    }

    pub fn set(
        self: *@This(),
        alloc: std.mem.Allocator,
        var_name: []const u8,
        val: RegId,
    ) Error!void {
        std.debug.assert(self.scopes.items.len != 0);

        const scope = &self.scopes.items[self.scopes.items.len - 1];
        try scope.put(alloc, var_name, val);
    }

    pub fn get(
        self: *@This(),
        var_name: []const u8,
    ) ?RegId {
        std.debug.assert(self.scopes.items.len != 0);

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
        try self.scopes.append(alloc, .{});
    }

    pub fn pop(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) void {
        var scope = self.scopes.pop().?;
        scope.deinit(alloc);
    }
};

pub const Builder = struct {
    scopes: std.ArrayList(std.MultiArrayList(Instr)) = .{},
    top: usize = 0,

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

    fn topScope(
        self: *@This(),
    ) *std.MultiArrayList(Instr) {
        return &self.scopes.items[self.top];
    }

    pub fn pushPrepare(
        self: *@This(),
        alloc: std.mem.Allocator,
        n: usize,
    ) Error!void {
        try self.lazyinit(alloc);
        try self.topScope().ensureUnusedCapacity(alloc, n);
    }

    pub fn push(
        self: *@This(),
        insrt: Instr,
    ) void {
        self.topScope().appendAssumeCapacity(insrt);
    }

    pub fn pushOne(
        self: *@This(),
        alloc: std.mem.Allocator,
        insrt: Instr,
    ) Error!void {
        try self.lazyinit(alloc);
        try self.topScope().append(alloc, insrt);
    }

    pub fn pushBlock(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) Error!void {
        try self.lazyinit(alloc);
        self.top += 1;
        try self.scopes.append(alloc, .{});
    }

    pub fn popBlock(
        self: *@This(),
        alloc: std.mem.Allocator,
        output: *std.MultiArrayList(Instr),
    ) Error!InstrRange {
        const top_scope = self.topScope();

        const instr: InstrRange = .{
            .start = .{@intCast(output.len)},
            .end = .{@intCast(output.len + top_scope.len)},
        };
        try output.ensureUnusedCapacity(alloc, top_scope.len);
        for (0..top_scope.len) |i| {
            output.appendAssumeCapacity(top_scope.get(i));
        }

        top_scope.clearRetainingCapacity();
        self.top -= 1;

        return instr;
    }
};
