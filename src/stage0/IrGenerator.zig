const std = @import("std");
const Parser = @import("Parser.zig");
const Tokenizer = @import("Tokenizer.zig");
const Span = Tokenizer.Span;
const Node = Parser.Node;
const NodeId = Parser.NodeId;

pub const NameHint = struct {
    prev: ?*const NameHint = null,
    part: []const u8 = "??",

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
pub const RegId = struct { u32 };

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
    /// create and initialize a meta variable: result
    let_result: struct {
        value: RegId,
    },
    /// reads the address of a field `a.b`
    get_field: struct {
        container: RegId,
        field_idx: u32,
    },
    @"return": struct {
        val: RegId,
    },

    str_lit: struct {
        val: Span,
    },
    int_lit: struct {
        val: u128,
    },
    float_lit: struct {
        val: f64,
    },
    void_lit: void,

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
        /// function to be called
        func: RegId,
    },
    /// calls a function `a()` with arguments given by earlier `.call_arg`s
    call_finish_builtin: struct {
        /// function to be called
        func: BuiltinFunc,
    },
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

pub const Static = struct {};

pub const Prototype = struct {
    // param_names: []const []const u8,
    // param_types: []u32,
    // return_type: u32,

    /// meta block is used to evaluate parameter and return types
    meta_block: InstrId,
};

pub const Function = struct {
    symbol: []const u8,
    /// meta block is used to evaluate parameter and return types
    meta_block: InstrId,
    /// main block is the actual code, before monomorphizing
    main_block: InstrId,
};

pub const Namespace = struct {};

pub const Error = error{
    OutOfMemory,
    UnknownSymbol,
};

instr: std.MultiArrayList(Instr) = .{},
// statics: std.MultiArrayList(Static) = .{},
functions: std.MultiArrayList(Function) = .{},
// prototypes: std.MultiArrayList(Prototype) = .{},
// namespaces: std.MultiArrayList(Namespace) = .{},
next_reg_id: RegId = .{1},
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

fn nextReg(
    self: *@This(),
) RegId {
    defer self.next_reg_id.@"0" += 1;
    return self.next_reg_id;
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!void {
    // measure the avg ir instruction count
    try self.instr.ensureTotalCapacity(alloc, 16);

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
    std.debug.print("FUNCTIONS:\n", .{});
    for (0..self.functions.len) |i| {
        const function = self.functions.get(i);
        std.debug.print(
            \\function {s}:
            \\  setup:
            \\    {}
            \\  entry:
            \\    {}
            \\
        , .{
            function.symbol,
            function.meta_block.@"0",
            function.main_block.@"0",
        });
    }
    std.debug.print("INSTR:\n", .{});
    for (0..self.instr.len) |i| {
        const instr = self.instr.get(i);
        std.debug.print("{any}\n", .{instr});
    }
}

pub fn convertStructContents(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    struct_contents: @FieldType(Node, "struct_contents"),
) Error!void {
    for (struct_contents.decls.start..struct_contents.decls.end) |i| {
        try self.convertDecl(alloc, name_hint, @intCast(i));
    }
}

pub fn convertDecl(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const decl = self.nodes()[node_id].decl;

    if (decl.type_hint) |type_hint| {
        _ = type_hint;
        @panic("todo");
        // const type_hint = try self.convertExpr(alloc, &.{
        //     .prev = name_hint,
        //     .part = decl.ident.read(self.parser.tokenizer.source),
        // }, type_hint);
    }

    const name = decl.ident.read(self.parser.tokenizer.source);
    const val = try self.convertExpr(alloc, &.{
        .prev = name_hint,
        .part = decl.ident.read(self.parser.tokenizer.source),
    }, decl.expr);

    try self.symbols.set(alloc, name, val);

    // try self.instr.append(alloc, .{ .let = .{
    //     .ident = decl.ident,
    //     .mut = decl.mut,
    //     .value = val,
    // } });
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

    const prototype = try self.convertProtoBare(alloc, name_hint, func.proto);
    const function = try self.convertFnBare(alloc, name_hint, node_id, prototype);

    try self.functions.append(alloc, function);
    // _ = function;

    return self.nextReg();
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
    errdefer self.builder.popBlockDiscard();

    const scope = try self.convertScope(alloc, name_hint, func.scope_or_symexpr);

    try self.builder.pushOne(alloc, .{ .@"return" = .{
        .val = scope,
    } });

    const main_block = try self.builder.popBlock(alloc, &self.instr);
    return .{
        .symbol = try name_hint.generate(alloc),
        .meta_block = prototype.meta_block,
        .main_block = main_block,
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
    errdefer self.builder.popBlockDiscard();

    for (proto.params.start..proto.params.end) |i| {
        const param = self.nodes()[i].param;
        const param_name = param.ident.read(self.parser.tokenizer.source);
        const param_type = try self.convertExpr(alloc, name_hint, param.type);

        try self.symbols.set(alloc, param_name, param_type);
    }

    if (proto.return_ty_expr) |i| {
        const return_ty = try self.convertExpr(alloc, name_hint, i);
        try self.builder.pushOne(alloc, .{ .let_result = .{
            .value = return_ty,
        } });
    }

    const meta_block = try self.builder.popBlock(alloc, &self.instr);
    return .{
        .meta_block = meta_block,
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
            self.builder.push(.{ .call_arg = .{
                .value = lhs,
            } });
            self.builder.push(.{ .call_arg = .{
                .value = rhs,
            } });
            self.builder.push(.{ .call_finish_builtin = .{
                .func = switch (v.op) {
                    inline else => |op| @field(BuiltinFunc, @tagName(op)),
                },
            } });

            return self.nextReg();
        },
        .unary_op => |v| {
            const val = try self.convertExpr(alloc, name_hint, v.val);

            try self.builder.pushPrepare(alloc, 2);
            self.builder.push(.{ .call_arg = .{
                .value = val,
            } });
            self.builder.push(.{ .call_finish_builtin = .{
                .func = switch (v.op) {
                    inline else => |op| @field(BuiltinFunc, @tagName(op)),
                },
            } });

            return self.nextReg();
        },
        .call => |v| {
            const func = try self.convertExpr(alloc, name_hint, v.val);

            const arg_regs = try alloc.alloc(RegId, v.args.len());
            defer alloc.free(arg_regs);

            for (arg_regs, v.args.start..v.args.end) |*arg, i| {
                arg.* = try self.convertExpr(alloc, name_hint, @intCast(i));
            }

            try self.builder.pushPrepare(alloc, @intCast(v.args.len() + 1));
            for (arg_regs) |arg| {
                self.builder.push(.{ .call_arg = .{
                    .value = arg,
                } });
            }
            self.builder.push(.{ .call_finish = .{
                .func = func,
            } });

            return self.nextReg();
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
            var val = v.tok;
            val.start += 1;
            val.end -= 1;

            try self.builder.pushOne(alloc, .{ .str_lit = .{
                .val = val,
            } });
            return self.nextReg();
        },
        .char_lit => |v| {
            try self.builder.pushOne(alloc, .{ .int_lit = .{
                .val = v.val,
            } });
            return self.nextReg();
        },
        .float_lit => |v| {
            try self.builder.pushOne(alloc, .{ .float_lit = .{
                .val = v.val,
            } });
            return self.nextReg();
        },
        .int_lit => |v| {
            try self.builder.pushOne(alloc, .{ .int_lit = .{
                .val = v.val,
            } });
            return self.nextReg();
        },
        .access => |v| {
            const sym = v.ident.read(self.parser.tokenizer.source);
            return self.symbols.get(sym) orelse {
                std.debug.print("unknown symbol: {s}\n", .{sym});
                return error.UnknownSymbol;
            };
        },
        .scope => return try self.convertScope(alloc, &.{
            .prev = name_hint,
            .part = "<scope>",
        }, node_id),
        .@"fn" => return try self.convertFn(alloc, name_hint, node_id),
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
        try self.builder.pushOne(alloc, .void_lit);
        return self.nextReg();
    };

    for (stmts.start..stmts.end) |i| {
        try self.convertStmt(alloc, name_hint, @intCast(i));
    }

    if (scope.has_trailing_semi) {
        return try self.convertExpr(alloc, name_hint, result_expr);
    } else {
        try self.convertStmt(alloc, name_hint, result_expr);

        try self.builder.pushOne(alloc, .void_lit);
        return self.nextReg();
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
            try self.convertDecl(alloc, name_hint, node_id);
        },
        else => {
            _ = try self.convertExpr(alloc, name_hint, node_id);
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
        val: RegId,
    ) Error!void {
        try self.lazyinit(alloc);

        const scope = &self.scopes.items[self.scopes.items.len - 1];
        try scope.put(alloc, var_name, val);
    }

    pub fn get(
        self: *@This(),
        var_name: []const u8,
    ) ?RegId {
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
    ) Error!InstrId {
        const top_scope = self.topScope();

        const instr_id: InstrId = .{@intCast(output.len)};
        try output.ensureUnusedCapacity(alloc, top_scope.len);
        for (0..top_scope.len) |i| {
            output.appendAssumeCapacity(top_scope.get(i));
        }

        top_scope.clearRetainingCapacity();
        self.top -= 1;

        return instr_id;
    }

    pub fn popBlockDiscard(
        self: *@This(),
    ) void {
        const top_scope = self.topScope();
        top_scope.clearRetainingCapacity();
        self.top -= 1;
    }
};
