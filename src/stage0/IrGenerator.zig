const std = @import("std");
const Parser = @import("Parser.zig");
const Tokenizer = @import("Tokenizer.zig");
const Span = Tokenizer.Span;
const Node = Parser.Node;
const NodeId = Parser.NodeId;
const UnaryOp = Parser.UnaryOp;
const BinaryOp = Parser.BinaryOp;
const Range = @import("main.zig").Range;
const NameHint = @import("main.zig").NameHint;
const log = std.log.scoped(.irgen);

//

pub const Value = enum(u32) {
    type_type,
    void_type,
    bool_type,
    u8_type,
    u16_type,
    u32_type,
    u64_type,
    usize_type,
    i8_type,
    i16_type,
    i32_type,
    i64_type,
    isize_type,
    f32_type,
    f64_type,
    c_int_type,
    c_char_type,
    c_long_type,
    c_longlong_type,
    c_short_type,
    c_uint_type,
    c_ulong_type,
    c_ulonglong_type,
    c_ushort_type,
    void,
    false,
    true,
    undefined,
    _,

    pub const builtin_count = @intFromEnum(Value.undefined) + 1;

    pub fn asIndex(self: @This()) ?Instr.Id {
        return switch (self) {
            _ => @enumFromInt(@intFromEnum(self) - builtin_count),
            else => null,
        };
    }

    pub fn format(self: *const @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
        if (self.asIndex()) |instr| {
            try writer.print("%{}", .{@intFromEnum(instr)});
        } else {
            try writer.print("@{t}", .{self.*});
        }
    }
};

pub const BuiltinVariable = enum {
    type,
    void,
    bool,
    u8,
    u16,
    u32,
    u64,
    usize,
    i8,
    i16,
    i32,
    i64,
    isize,
    f32,
    f64,

    c_int,
    c_char,
    c_long,
    c_longlong,
    c_short,
    c_uint,
    c_ulong,
    c_ulonglong,
    c_ushort,

    false,
    true,
    undefined,
};

/// index into the `extras` array
pub const Extra = enum(u32) {
    _,

    pub fn advance(
        self: @This(),
        comptime T: type,
    ) @This() {
        return @enumFromInt(@intFromEnum(self) + size(T));
    }

    pub fn size(
        comptime T: type,
    ) comptime_int {
        return @divExact(@sizeOf(T), @sizeOf(u32));
    }

    pub fn addProto(
        extras: *std.ArrayList(u32),
        alloc: std.mem.Allocator,
        argc: usize,
    ) error{OutOfMemory}!struct { Extra, *Proto, []Param } {
        const extra: Extra = @enumFromInt(extras.items.len);
        const proto = try extras.addManyAsArray(alloc, size(Proto));
        _, const params = try addParams(extras, alloc, argc);
        return .{ extra, @ptrCast(proto), @ptrCast(params) };
    }

    pub fn getProto(
        extras: []const u32,
        extra: Extra,
    ) struct { Proto, []const Param } {
        std.debug.assert(size(u32) == size(Param));
        const proto: Proto = @bitCast(extras[@intFromEnum(extra)..][0..size(Proto)].*);
        const params = getParams(
            extras,
            extra.advance(Proto),
            proto.param_count,
        );
        return .{ proto, params };
    }

    pub fn addParams(
        extras: *std.ArrayList(u32),
        alloc: std.mem.Allocator,
        argc: usize,
    ) error{OutOfMemory}!struct { Extra, []Param } {
        std.debug.assert(size(u32) == size(Param));
        const extra: Extra = @enumFromInt(extras.items.len);
        const params = try extras.addManyAsSlice(alloc, argc);
        return .{ extra, @ptrCast(params) };
    }

    pub fn getParams(
        extras: []const u32,
        extra: Extra,
        argc: usize,
    ) []const Param {
        return @ptrCast(extras[@intFromEnum(extra)..][0..argc]);
    }

    pub const Proto = extern struct {
        return_type: Value,
        /// number of `Param` following this
        /// `Proto` in the `extras` array
        param_count: u32,
    };

    pub const Param = extern struct {
        val: Value,
    };
};

pub const Instr = union(enum) {
    pub const Id = enum(u32) {
        start,
        _,

        pub fn asValue(self: @This()) Value {
            return @enumFromInt(@intFromEnum(self) + Value.builtin_count);
        }
    };

    str_lit: struct {
        value: Span,
    },
    int_lit: struct {
        value: u64,
    },
    float_lit: struct {
        value: f64,
    },
    // builtin_lit: struct {
    //     builtin: BuiltinVariable,
    // },
    call: struct {
        func: Value,
        argv: Extra,
        argc: u32,
    },
    unary_op: struct {
        value: Value,
        op: UnaryOp,
    },
    binary_op: struct {
        lhs: Value,
        rhs: Value,
        op: BinaryOp,
    },
    array: struct {
        len: Value,
        child: Value,
    },
    alloca: struct {
        ty: Value,
    },
    as: struct {
        ty: Value,
        val: Value,
    },
    /// creates a new function or a global
    /// only usable in a struct
    decl: struct {
        name: Span,
        block_end: Instr.Id,
    },
    /// creates a new anonymous function
    func: struct {
        proto: Value,
        body_block_end: Instr.Id,
    },
    /// creates a new function type
    proto: struct {
        /// index into the `extras` array
        extra: Extra,
    },
    /// saves a function parameter to a register
    param,
    /// in a struct: completes the struct
    /// in a proto block: declares the function return type and completes the proto
    /// in code: returns from a block with a value
    @"break": struct {
        block: Instr.Id,
        val: Value,
    },
    /// tells which source line:col the next instructions are from
    dbg_loc: struct {
        line: u32,
        col: u32,
    },
    /// tells which source variable name the value is from
    dbg_name: struct {
        name: Span,
        val: Value,
    },
    /// prints a value at compile time
    dbg_print: struct {
        val: Value,
    },
    /// a block of instructions which can return with a value
    block: struct {
        block_end: Instr.Id,
    },
    conditional: struct {
        boolean: Value,
        on_true_block_end: Instr.Id,
        on_false_block_end: Instr.Id,
    },
    // unconditional: struct {
    //     dst: Instr.Index,
    // },
};

pub const Error = error{
    TooManyRegisters,
    OutOfMemory,
    VariableNotFound,
    MainFunctionMissing,
};

instrs: std.MultiArrayList(Instr) = .{},
extras: std.ArrayList(u32) = .{},
symbols: Symbols = .{},
main: Value = .undefined,
// builder: Builder = .{},

// builder: Builder = .{},
// current_block: BlockId = .{ .i = 0 },
// string_arena: std.heap.ArenaAllocator = undefined,

// root_namespace: InstrId = 0,
parser: *Parser,

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    // self.builder.deinit(alloc);
    self.symbols.deinit(alloc);
    self.extras.deinit(alloc);
    self.instrs.deinit(alloc);
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

fn allocDebugName(
    self: *@This(),
    name: NameHint,
) Error![]const u8 {
    return try name.generate(self.string_arena.allocator());
}

fn pushScope(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!void {
    try self.symbols.pushScope(alloc);
}

fn popScope(
    self: *@This(),
) void {
    self.symbols.popScope();
}

fn nextInstr(
    self: *@This(),
) Instr.Id {
    return @enumFromInt(self.instrs.len);
}

fn pushInstr(
    self: *@This(),
    alloc: std.mem.Allocator,
    instr: Instr,
) Error!Instr.Id {
    const id = try self.instrs.addOne(alloc);
    self.instrs.set(id, instr);
    return @enumFromInt(id);
}

fn pushInstrGetValue(
    self: *@This(),
    alloc: std.mem.Allocator,
    instr: Instr,
) Error!Value {
    const instr_addr = try self.pushInstr(alloc, instr);
    return instr_addr.asValue();
}

fn overwriteInstr(self: *@This(), idx: Instr.Id, instr: Instr) void {
    self.instrs.set(@intFromEnum(idx), instr);
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!void {
    // self.string_arena = .init(alloc);

    // TODO: measure the avg ir instruction count per source token
    try self.instrs.ensureTotalCapacity(alloc, 16);

    try self.pushScope(alloc);
    defer self.popScope();

    try self.symbols.createVar(alloc, "u32", Value.u32_type);

    const root_name_hint: NameHint = .new("root");
    try self.convertStructContents(
        alloc,
        &root_name_hint,
        0,
    );

    self.main = self.symbols.findVar("main") orelse {
        return error.MainFunctionMissing;
    };
    _ = try self.pushInstr(alloc, .{ .call = .{
        .func = self.main,
        .argc = 0,
        .argv = @enumFromInt(0),
    } });
}

pub fn dump(
    self: *@This(),
) void {
    self.dumpBlock(@enumFromInt(0), 0);
    std.debug.print(";; instr extra = {}\n", .{self.extras.items.len});
    std.debug.print(";; instr count = {}\n", .{self.instrs.len});
    std.debug.print(";; main = {f}\n", .{self.main});
}

fn dumpBlock(
    self: *@This(),
    start: Instr.Id,
    indent: usize,
) void {
    var cur = start;

    while (@intFromEnum(cur) < self.instrs.len) {
        const instr = self.instrs.get(@intFromEnum(cur));
        for (0..indent) |_| std.debug.print("    ", .{});
        std.debug.print("%{} = ", .{@intFromEnum(cur)});
        cur = @enumFromInt(@intFromEnum(cur) + 1);

        switch (instr) {
            .str_lit => |v| {
                std.debug.print("str_lit(\"{s}\")\n", .{v.value.read(self.source())});
            },
            .int_lit => |v| {
                std.debug.print("int_lit({})\n", .{v.value});
            },
            .float_lit => |v| {
                std.debug.print("float_lit({})\n", .{v.value});
            },
            .call => |v| {
                const args = Extra.getParams(
                    self.extras.items,
                    v.argv,
                    v.argc,
                );
                std.debug.print("call(func={f}, args=[", .{v.func});
                for (args) |arg| {
                    std.debug.print("{f}, ", .{arg.val});
                }
                std.debug.print("])\n", .{});
            },
            .unary_op => |v| {
                std.debug.print("unary_op(op={f}, value={f})\n", .{ v.op, v.value });
            },
            .binary_op => |v| {
                std.debug.print("binary_op(op={f}, lhs={f}, rhs={f})\n", .{
                    v.op, v.lhs, v.rhs,
                });
            },
            .array => |v| {
                std.debug.print("array(len={f}, child={f})\n", .{
                    v.len, v.child,
                });
            },
            .alloca => |v| {
                std.debug.print("alloca(type={f})\n", .{v.ty});
            },
            .as => |v| {
                std.debug.print("as(type={f}, value={f})\n", .{ v.ty, v.val });
            },
            .decl => |v| {
                std.debug.print("decl(name=\"{s}\", block={{\n", .{v.name.read(self.source())});
                self.dumpBlock(cur, indent + 1);
                for (0..indent) |_| std.debug.print("    ", .{});
                std.debug.print("}})\n", .{});
                cur = v.block_end;
            },
            .func => |v| {
                std.debug.print("func(proto={f}, body={{\n", .{v.proto});
                self.dumpBlock(cur, indent + 1);
                for (0..indent) |_| std.debug.print("    ", .{});
                std.debug.print("}})\n", .{});
                cur = v.body_block_end;
            },
            .proto => |v| {
                const extra, const params = Extra.getProto(
                    self.extras.items,
                    v.extra,
                );

                std.debug.print("proto(return_type={f}, params=[", .{
                    extra.return_type,
                });
                if (params.len != 0) {
                    std.debug.print("{f}", .{params[0].val});
                    for (params[1..]) |param| {
                        std.debug.print(", {f}", .{param.val});
                    }
                }
                std.debug.print("])\n", .{});
            },
            .param => {
                std.debug.print("param\n", .{});
            },
            .@"break" => |v| {
                std.debug.print("break(block=%{}, value={f})\n", .{ @intFromEnum(v.block), v.val });
                return;
            },
            .dbg_loc => |v| {
                std.debug.print("dbg_loc(line={}, col={})\n", .{ v.line, v.col });
            },
            .dbg_name => |v| {
                std.debug.print("dbg_name(name=\"{s}\", val={f})\n", .{
                    v.name.read(self.source()),
                    v.val,
                });
            },
            .dbg_print => |v| {
                std.debug.print("dbg_print(val={f})\n", .{v.val});
            },
            .block => |v| {
                std.debug.print("block(proto={{\n", .{});
                self.dumpBlock(cur, indent + 1);
                for (0..indent) |_| std.debug.print("    ", .{});
                std.debug.print("}})\n", .{});
                cur = v.block_end;
            },
            .conditional => |v| {
                std.debug.print("conditional(check={f}, on_true_block={{\n", .{v.boolean});
                self.dumpBlock(cur, indent + 1);
                for (0..indent) |_| std.debug.print("    ", .{});
                std.debug.print("}}, on_false_block={{\n", .{});
                self.dumpBlock(v.on_true_block_end, indent + 1);
                for (0..indent) |_| std.debug.print("    ", .{});
                std.debug.print("}})\n", .{});
                cur = v.on_false_block_end;
            },
        }
    }
}

pub fn convertStructContents(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const struct_contents = self.nodes()[node_id].struct_contents;

    for (struct_contents.decls.start..struct_contents.decls.end) |i| {
        try self.convertDecl(
            alloc,
            name_hint,
            @intCast(i),
        );
    }
}

pub fn convertDecl(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const decl = self.nodes()[node_id].decl;
    const name = decl.ident.read(self.source());
    const next_name_hint = name_hint.push(name);
    const init_name_hint = next_name_hint.push("init");

    var val = try self.convertExpr(
        alloc,
        &next_name_hint,
        decl.expr,
    );

    if (decl.type_hint) |type_hint| {
        const ty = try self.convertExpr(
            alloc,
            &init_name_hint,
            type_hint,
        );

        val = try self.pushInstrGetValue(alloc, .{ .as = .{
            .ty = ty,
            .val = val,
        } });
    }

    const named_val = try self.pushInstrGetValue(alloc, .{ .dbg_name = .{
        .name = decl.ident,
        .val = val,
    } });
    try self.symbols.createVar(
        alloc,
        name,
        named_val,
    );
}

pub fn convertExpr(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    switch (self.nodes()[node_id]) {
        .@"if" => return try self.convertIf(
            alloc,
            name_hint,
            node_id,
        ),
        .proto => return try self.convertProto(
            alloc,
            name_hint,
            node_id,
        ),
        .@"fn" => return try self.convertFn(
            alloc,
            name_hint,
            node_id,
        ),
        .scope => return try self.convertScope(
            alloc,
            name_hint,
            node_id,
        ),
        .array => return try self.convertArray(
            alloc,
            name_hint,
            node_id,
        ),
        .slice => return try self.convertSlice(
            alloc,
            name_hint,
            node_id,
        ),
        .pointer => return try self.convertPointer(
            alloc,
            name_hint,
            node_id,
        ),
        .binary_op => return try self.convertBinaryOp(
            alloc,
            name_hint,
            node_id,
        ),
        .field_acc => return try self.convertFieldAcc(
            alloc,
            name_hint,
            node_id,
        ),
        .call => return try self.convertCall(
            alloc,
            name_hint,
            node_id,
        ),
        .access => |acc| return try self.convertAccess(
            alloc,
            acc.ident.read(self.source()),
        ),
        .str_lit => |lit| return try self.convertStrLit(
            alloc,
            lit.tok,
        ),
        .float_lit => |lit| return try self.convertFloatLit(
            alloc,
            lit.val,
        ),
        .int_lit => |lit| return try self.convertIntLit(
            alloc,
            lit.val,
        ),
        else => std.debug.panic("TODO: {}", .{self.nodes()[node_id]}),
    }
}

pub fn convertAssign(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const assign = self.nodes()[node_id].assign;

    const target = try self.convertExpr(
        alloc,
        name_hint,
        assign.lhs,
    );

    const value = try self.convertExpr(
        alloc,
        name_hint,
        assign.rhs,
    );

    _ = target;
    _ = value;
    @panic("todo");

    // try self.pushInstr(alloc, .{ .assign = .{
    //     .target = target,
    //     .value = value,
    // } });
}

pub fn convertIf(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const @"if" = self.nodes()[node_id].@"if";

    const name_hint_check = name_hint.push("check");
    const name_hint_on_true = name_hint.push("on_true");
    const name_hint_on_false = name_hint.push("on_false");

    const if_block = try self.pushInstr(alloc, .{ .block = undefined });

    const boolean = try self.convertExpr(
        alloc,
        &name_hint_check,
        @"if".check_expr,
    );

    const conditional = try self.pushInstr(alloc, .{ .conditional = undefined });

    const on_true_val = try self.convertScope(
        alloc,
        &name_hint_on_true,
        @"if".on_true_scope,
    );
    _ = try self.pushInstr(alloc, .{ .@"break" = .{
        .block = if_block,
        .val = on_true_val,
    } });
    const on_true_block_end = self.nextInstr();

    const on_false_val = try self.convertScope(
        alloc,
        &name_hint_on_false,
        @"if".on_false_scope,
    );
    _ = try self.pushInstr(alloc, .{ .@"break" = .{
        .block = if_block,
        .val = on_false_val,
    } });
    const on_false_block_end = self.nextInstr();

    self.instrs.set(@intFromEnum(conditional), .{ .conditional = .{
        .boolean = boolean,
        .on_true_block_end = on_true_block_end,
        .on_false_block_end = on_false_block_end,
    } });
    self.instrs.set(@intFromEnum(if_block), .{ .block = .{
        .block_end = on_false_block_end,
    } });

    return if_block.asValue();
}

pub fn convertProto(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const proto = self.nodes()[node_id].proto;

    try self.pushScope(alloc);
    defer self.popScope();

    const extra, const proto_extra, const params_extra = try Extra.addProto(
        &self.extras,
        alloc,
        proto.params.len(),
    );

    const name_hint_proto = name_hint.push("proto");
    const name_hint_param = name_hint_proto.push("param");
    for (0..proto.params.len()) |i| {
        const param = self.nodes()[proto.params.start + i].param;
        const param_name = param.ident.read(self.source());
        const param_type = try self.convertExpr(
            alloc,
            &name_hint_param.push(param_name),
            param.type,
        );
        const param_value = try self.pushInstrGetValue(alloc, .{ .as = .{
            .ty = param_type,
            .val = .undefined,
        } });

        try self.symbols.createVar(alloc, param_name, param_value);
        params_extra[i] = .{ .val = param_type };
    }

    const return_type = if (proto.return_ty_expr) |expr_node_id| b: {
        const name_hint_ret = name_hint_proto.push("ret");
        break :b try self.convertExpr(
            alloc,
            &name_hint_ret,
            expr_node_id,
        );
    } else Value.void_type;

    proto_extra.* = .{
        .param_count = proto.params.len(),
        .return_type = return_type,
    };

    return self.pushInstrGetValue(alloc, .{ .proto = .{
        .extra = extra,
    } });
}

pub fn convertFn(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const func_node = self.nodes()[node_id].@"fn";
    const proto_node = self.nodes()[func_node.proto].proto;

    try self.pushScope(alloc);
    defer self.popScope();

    const proto = try self.convertProto(
        alloc,
        name_hint,
        func_node.proto,
    );

    const func_block = try self.pushInstr(alloc, .{ .func = undefined });
    const func_block_start = self.nextInstr();

    const name_hint_fn = name_hint.push(if (proto_node.@"extern") "symexpr" else "fn");
    if (proto_node.@"extern") {
        const symbol = try self.convertExpr(
            alloc,
            &name_hint_fn,
            func_node.scope_or_symexpr,
        );
        _ = try self.pushInstr(alloc, .{ .@"break" = .{
            .block = func_block_start,
            .val = symbol,
        } });
    } else {
        for (proto_node.params.start..proto_node.params.end) |_| {
            _ = try self.pushInstrGetValue(alloc, .param);
        }

        for (proto_node.params.start..proto_node.params.end) |i| {
            const param = self.nodes()[i].param;
            const val: Value = @enumFromInt(@intFromEnum(func_block_start.asValue()) + 1);
            const named_val = try self.pushInstrGetValue(alloc, .{ .dbg_name = .{
                .name = param.ident,
                .val = val,
            } });
            try self.symbols.createVar(
                alloc,
                param.ident.read(self.source()),
                named_val,
            );
        }

        const return_value = try self.convertScope(
            alloc,
            &name_hint_fn,
            func_node.scope_or_symexpr,
        );
        _ = try self.pushInstr(alloc, .{ .@"break" = .{
            .block = func_block_start,
            .val = return_value,
        } });
    }
    const body_block_end = self.nextInstr();

    self.overwriteInstr(func_block, .{ .func = .{
        .proto = proto,
        .body_block_end = body_block_end,
    } });
    return func_block.asValue();
}

pub fn convertScope(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const scope = self.nodes()[node_id].scope;

    const stmts, const last_stmt = scope.stmts.splitLast() orelse {
        return Value.void;
    };

    try self.pushScope(alloc);
    defer self.popScope();

    for (stmts.start..stmts.end) |stmt| {
        try self.convertStmt(
            alloc,
            name_hint,
            @intCast(stmt),
        );
    }

    if (scope.has_trailing_semi) {
        try self.convertStmt(
            alloc,
            name_hint,
            last_stmt,
        );
        return Value.void;
    } else {
        return try self.convertExpr(
            alloc,
            name_hint,
            last_stmt,
        );
    }
}

pub fn convertStmt(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    switch (self.nodes()[node_id]) {
        .decl => try self.convertDecl(
            alloc,
            name_hint,
            node_id,
        ),
        .assign => try self.convertAssign(
            alloc,
            name_hint,
            node_id,
        ),
        else => _ = try self.convertExpr(
            alloc,
            name_hint,
            node_id,
        ),
    }
}

pub fn convertArray(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const array = self.nodes()[node_id].array;

    const length = try self.convertExpr(
        alloc,
        &name_hint.push("length"),
        array.length_expr,
    );
    const element = try self.convertExpr(
        alloc,
        &name_hint.push("element"),
        array.elements_expr,
    );

    return try self.pushInstrGetValue(
        alloc,
        .{ .array = .{
            .len = length,
            .child = element,
        } },
    );
}

pub fn convertSlice(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const slice = self.nodes()[node_id].slice;

    const elements = try self.convertExpr(
        alloc,
        name_hint,
        slice.elements_expr,
    );

    const result = try self.pushInstr(alloc, .{ .unary_op = .{
        .value = elements,
        .op = if (slice.mut) .slice_mut else .slice,
    } });
    return result.asValue();
}

pub fn convertPointer(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const pointer = self.nodes()[node_id].pointer;

    const elements = try self.convertExpr(
        alloc,
        name_hint,
        pointer.pointee_expr,
    );

    const result = try self.pushInstr(alloc, .{ .unary_op = .{
        .value = elements,
        .op = if (pointer.mut) .pointer_mut else .pointer,
    } });
    return result.asValue();
}

pub fn convertBinaryOp(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const binary_op = self.nodes()[node_id].binary_op;

    const lhs = try self.convertExpr(
        alloc,
        name_hint,
        binary_op.lhs,
    );

    const rhs = try self.convertExpr(
        alloc,
        name_hint,
        binary_op.rhs,
    );

    const result = try self.pushInstr(alloc, .{ .binary_op = .{
        .lhs = lhs,
        .rhs = rhs,
        .op = binary_op.op,
    } });
    return result.asValue();
}

pub fn convertFieldAcc(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const field_acc = self.nodes()[node_id].field_acc;

    const container = try self.convertExpr(
        alloc,
        name_hint,
        field_acc.val,
    );

    const field = try self.pushInstrGetValue(alloc, .{
        .str_lit = .{ .value = field_acc.ident },
    });

    const result = try self.pushInstr(alloc, .{ .binary_op = .{
        .lhs = container,
        .rhs = field,
        .op = BinaryOp.field,
    } });
    return result.asValue();
}

pub fn convertCall(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const call = self.nodes()[node_id].call;

    const argc: u32 = call.args.len();
    const argv, const args = try Extra.addParams(
        &self.extras,
        alloc,
        argc,
    );

    for (call.args.start..call.args.end, args) |expr_node_id, *arg| {
        const arg_expr_result = try self.convertExpr(
            alloc,
            name_hint,
            @intCast(expr_node_id),
        );
        arg.* = .{ .val = arg_expr_result };
    }

    const func = try self.convertExpr(
        alloc,
        name_hint,
        call.val,
    );

    const result = try self.pushInstr(alloc, .{ .call = .{
        .func = func,
        .argv = argv,
        .argc = argc,
    } });
    return result.asValue();
}

pub fn convertAccess(
    self: *@This(),
    alloc: std.mem.Allocator,
    var_name: []const u8,
) Error!Value {
    _ = alloc;
    // if (std.mem.eql(u8, "_", var_name)) {
    //     const result = self.registers.pushTmp();
    //     //
    //     return result;
    // }

    if (std.meta.stringToEnum(BuiltinVariable, var_name)) |builtin| {
        return switch (builtin) {
            .type => Value.type_type,
            .void => Value.void_type,
            .bool => Value.bool_type,
            .u8 => Value.u8_type,
            .u16 => Value.u16_type,
            .u32 => Value.u32_type,
            .u64 => Value.u64_type,
            .usize => Value.usize_type,
            .i8 => Value.i8_type,
            .i16 => Value.i16_type,
            .i32 => Value.i32_type,
            .i64 => Value.i64_type,
            .isize => Value.isize_type,
            .f32 => Value.f32_type,
            .f64 => Value.f64_type,

            .c_int => Value.c_int_type,
            .c_char => Value.c_char_type,
            .c_long => Value.c_long_type,
            .c_longlong => Value.c_longlong_type,
            .c_short => Value.c_short_type,
            .c_uint => Value.c_uint_type,
            .c_ulong => Value.c_ulong_type,
            .c_ulonglong => Value.c_ulonglong_type,
            .c_ushort => Value.c_ushort_type,

            .false => Value.false,
            .true => Value.true,
            .undefined => Value.undefined,
        };
    }

    const result = self.symbols.findVar(var_name) orelse {
        log.debug("variable not found: {s}", .{var_name});
        return Error.VariableNotFound;
    };
    return result;
}

pub fn convertStrLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    span: Span,
) Error!Value {
    const contents = span.read(self.source());

    std.debug.assert(span.len() >= 2);
    std.debug.assert(contents[0] == '"');
    std.debug.assert(contents[contents.len - 1] == '"');

    var span_without_quotes = span;
    span_without_quotes.start += 1;
    span_without_quotes.end -= 1;

    return try self.pushInstrGetValue(alloc, .{ .str_lit = .{
        .value = span_without_quotes,
    } });
}

pub fn convertFloatLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    value: f64,
) Error!Value {
    const result = try self.pushInstr(alloc, .{ .float_lit = .{
        .value = value,
    } });
    return result.asValue();
}

pub fn convertIntLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    value: u128,
) Error!Value {
    // TODO: support big ints
    const result = try self.pushInstr(alloc, .{ .int_lit = .{
        .value = @intCast(value),
    } });
    return result.asValue();
}

pub const Symbols = struct {
    var_name_hashmap: std.StringHashMapUnmanaged(ShadowChainEntry) = .{},
    shadow_chain: std.ArrayList(ShadowChainEntry) = .{},
    val_names: std.AutoHashMapUnmanaged(Value, []const u8) = .{},

    /// holds the number of values in the value stack at that scopes position
    scope_sizes: std.ArrayList(u32) = .{},
    /// holds all values of all scopes
    value_stack: std.ArrayList(Value) = .{},

    const ShadowChainEntry = struct {
        prev: u32,
        this: Value,
    };

    pub fn deinit(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) void {
        self.value_stack.deinit(alloc);
        self.scope_sizes.deinit(alloc);
        self.val_names.deinit(alloc);
        self.shadow_chain.deinit(alloc);
        self.var_name_hashmap.deinit(alloc);
    }

    pub fn createVar(
        self: *@This(),
        alloc: std.mem.Allocator,
        var_name: []const u8,
        val: Value,
    ) Error!void {
        try self.var_name_hashmap.ensureUnusedCapacity(alloc, 1);
        try self.shadow_chain.ensureUnusedCapacity(alloc, 1);
        try self.val_names.ensureUnusedCapacity(alloc, 1);
        try self.value_stack.ensureUnusedCapacity(alloc, 1);
        std.debug.assert(self.scope_sizes.items.len != 0);

        self.scope_sizes.items[self.scope_sizes.items.len - 1] += 1;
        self.value_stack.appendAssumeCapacity(val);

        self.val_names.putAssumeCapacity(val, var_name);
        const lookup_entry = self.var_name_hashmap.getOrPutAssumeCapacity(var_name);
        if (lookup_entry.found_existing) {
            const prev: u32 = @intCast(self.shadow_chain.items.len);
            self.shadow_chain.appendAssumeCapacity(lookup_entry.value_ptr.*);
            lookup_entry.value_ptr.* = .{
                .prev = prev,
                .this = val,
            };
        } else {
            lookup_entry.value_ptr.* = .{
                .prev = std.math.maxInt(u32),
                .this = val,
            };
        }
    }

    pub fn findVar(
        self: *@This(),
        var_name: []const u8,
    ) ?Value {
        const entry = self.var_name_hashmap.get(var_name) orelse return null;
        return entry.this;
    }

    fn popVar(
        self: *@This(),
        var_name: []const u8,
    ) void {
        const lookup_entry = self.var_name_hashmap.getPtr(var_name) orelse return;

        if (lookup_entry.prev != std.math.maxInt(u32)) {
            const tmp = self.shadow_chain.items[lookup_entry.prev];
            self.shadow_chain.items[lookup_entry.prev] = undefined;
            lookup_entry.* = tmp;
        } else {
            std.debug.assert(self.var_name_hashmap.remove(var_name));
        }
    }

    pub fn pushScope(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) Error!void {
        try self.scope_sizes.append(alloc, 0);
    }

    pub fn popScope(
        self: *@This(),
    ) void {
        const values = self.scope_sizes.pop().?;

        for (0..values) |_| {
            const val = self.value_stack.pop().?;
            const val_name = (self.val_names.fetchRemove(val) orelse continue).value;
            self.popVar(val_name);
        }
    }
};

// pub const Builder = struct {
//     instrs: std.ArrayList(Instr) = .{},
//     block_instr_counts: std.ArrayList(u32) = .{},
//     top_block_instr_count: u32 = 0,

//     pub fn deinit(
//         self: *@This(),
//         alloc: std.mem.Allocator,
//     ) void {
//         self.block_instr_counts.deinit(alloc);
//         self.instrs.deinit(alloc);
//     }

//     pub fn pushInstr(
//         self: *@This(),
//         alloc: std.mem.Allocator,
//         instr: Instr,
//     ) Error!void {
//         try self.instrs.append(alloc, instr);
//         self.top_block_instr_count += 1;
//     }

//     pub fn pushBlock(
//         self: *@This(),
//         alloc: std.mem.Allocator,
//     ) Error!void {
//         try self.block_instr_counts.append(alloc, self.top_block_instr_count);
//         self.top_block_instr_count = 0;
//     }

//     pub fn popBlock(
//         self: *@This(),
//         alloc: std.mem.Allocator,
//         instrs_output: *std.MultiArrayList(Instr),
//     ) Error!Instr.Index {
//         const instrs = self.top_block_instr_count;
//         self.top_block_instr_count = self.block_instr_counts.pop() orelse 0;

//         // const instr: InstrRange = .{
//         //     .start = .{ .i = @intCast(instrs_output.items.len) },
//         //     .end = .{ .i = @intCast(instrs_output.items.len + instrs) },
//         // };
//         const start: Instr.Index = @enumFromInt(instrs_output.len);
//         try instrs_output.ensureUnusedCapacity(alloc, instrs);
//         instrs_output.len += instrs;
//         for (self.instrs.items, instrs..) |instr, i| {
//             instrs_output.set(i, instr);
//         }
//         self.instrs.items.len -= instrs;
//         return start;
//     }
// };
