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

pub const RegId = struct {
    i: u32,
    pub fn format(self: *const @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
        return writer.print("%{}", .{self.i});
    }
};
pub const InstrId = struct {
    i: u32,
    pub fn format(self: *const @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
        return writer.print("i{}", .{self.i});
    }
};
pub const BlockId = struct {
    i: u32,
    pub fn format(self: *const @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
        return writer.print("b{}", .{self.i});
    }
};
pub const StructId = struct {
    i: u32,
    pub fn format(self: *const @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
        return writer.print("s{}", .{self.i});
    }
};
pub const ProtoId = struct {
    i: u32,
    pub fn format(self: *const @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
        return writer.print("p{}", .{self.i});
    }
};
pub const FunctionId = struct {
    i: u32,
    pub fn format(self: *const @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
        return writer.print("f{}", .{self.i});
    }
};
pub const InstrRange = Range(InstrId, .{ .i = 0 });

pub const Instr = union(enum) {
    str_lit: struct {
        result: RegId,
        value: []const u8,
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
    builtin_lit: struct {
        result: RegId,
        builtin: BuiltinVariable,
    },
    call: struct {
        result: RegId,
        func: RegId,
    },
    unary_op: struct {
        result: RegId,
        value: RegId,
        op: UnaryOp,
    },
    binary_op: struct {
        result: RegId,
        lhs: RegId,
        rhs: RegId,
        op: BinaryOp,
    },
    arg_fetch: struct {
        result: RegId,
    },
    assign: struct {
        target: RegId,
        value: RegId,
    },
    decl_fn: struct {
        result: RegId,
        func: FunctionId,
    },
    decl_proto: struct {
        result: RegId,
        proto: ProtoId,
    },
    decl_param: struct {
        ty: RegId,
    },
    decl_return: struct {
        ty: RegId,
    },
    decl_arg: struct {
        value: RegId,
    },
};

pub const BranchInstr = union(enum) {
    conditional: struct {
        boolean: RegId,
        on_true: BlockId,
        on_false: BlockId,
    },
    unconditional: BlockId,
    ret: RegId,
    end,
};

pub const Block = struct {
    debug_name: []const u8,
    instructions: InstrRange,
    branch_instruction: BranchInstr,
};

pub const Struct = struct {
    debug_name: []const u8,
    // parent: StructId,
    decl_block: BlockId,
};

pub const Function = struct {
    debug_name: []const u8,
    // parent: StructId,
    proto: ProtoId,
    entry_block: BlockId,
};

pub const Proto = struct {
    debug_name: []const u8,
    // parent: StructId,
    is_extern: bool,
    is_va_args: bool,
    decl_block: BlockId,
};

pub const Error = error{
    TooManyRegisters,
    OutOfMemory,
    VariableNotFound,
    MainFunctionMissing,
};

pub const BuiltinVariable = enum {
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
    bool,
    void,
    auto_int,
    auto_float,

    c_int,
    c_char,
    c_long,
    c_longdouble,
    c_longlong,
    c_short,
    c_uint,
    c_ulong,
    c_ulonglong,
    c_ushort,

    false,
    true,
};

instrs: std.ArrayList(Instr) = .{},
blocks: std.ArrayList(Block) = .{},
structs: std.ArrayList(Struct) = .{},
protos: std.ArrayList(Proto) = .{},
functions: std.ArrayList(Function) = .{},

param_stack: std.ArrayList(RegId) = .{},
registers: Registers = .{},
builder: Builder = .{},
current_block: BlockId = .{ .i = 0 },
string_arena: std.heap.ArenaAllocator = undefined,

// root_namespace: InstrId = 0,
parser: *Parser,

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    self.string_arena.deinit();

    self.builder.deinit(alloc);
    self.registers.deinit(alloc);
    self.param_stack.deinit(alloc);

    self.functions.deinit(alloc);
    self.protos.deinit(alloc);
    self.structs.deinit(alloc);
    self.blocks.deinit(alloc);
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

fn allocBlock(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!BlockId {
    const block_id: BlockId = .{ .i = @intCast(self.blocks.items.len) };
    try self.blocks.append(alloc, undefined);
    return block_id;
}

fn allocStruct(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!StructId {
    const struct_id: StructId = .{ .i = @intCast(self.structs.items.len) };
    try self.structs.append(alloc, undefined);
    return struct_id;
}

fn allocProto(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!ProtoId {
    const proto_id: ProtoId = .{ .i = @intCast(self.protos.items.len) };
    try self.protos.append(alloc, undefined);
    return proto_id;
}

fn allocFunction(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!FunctionId {
    const function_id: FunctionId = .{ .i = @intCast(self.functions.items.len) };
    try self.functions.append(alloc, undefined);
    return function_id;
}

fn pushScope(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!void {
    try self.registers.pushScope(alloc);
}

fn popScope(
    self: *@This(),
) void {
    self.registers.popScope();
}

fn pushBlock(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!BlockId {
    try self.builder.pushBlock(alloc);
    self.current_block = try self.allocBlock(alloc);
    return self.current_block;
}

fn popBlock(
    self: *@This(),
    alloc: std.mem.Allocator,
    branch_instr: BranchInstr,
    block_id: ?BlockId,
    name_hint: NameHint,
) Error!void {
    const current_block = block_id orelse self.current_block;
    self.blocks.items[current_block.i] = try self.builder.popBlock(
        alloc,
        &self.instrs,
        branch_instr,
        try self.allocDebugName(name_hint),
    );
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!void {
    self.string_arena = .init(alloc);

    // TODO: measure the avg ir instruction count per source token
    try self.instrs.ensureTotalCapacity(alloc, 16);
    // TODO: measure the avg ir block count per source token
    try self.blocks.ensureTotalCapacity(alloc, 16);
    // TODO: measure the avg ir struct count per source token
    try self.structs.ensureTotalCapacity(alloc, 16);
    // TODO: measure the avg ir function count per source token
    try self.functions.ensureTotalCapacity(alloc, 16);

    const root_name_hint: NameHint = .new("<root>");
    _ = try self.pushBlock(alloc);
    try self.convertStructContents(
        alloc,
        &root_name_hint,
        0,
    );

    // const main = self.registers.findVar("main") orelse {
    //     return Error.MainFunctionMissing;
    // };
    // try self.builder.pushInstr(alloc, .{ .decl_entrypoint = .{
    //     .func = main,
    // } });

    const ret = try self.convertVoidLit(alloc);
    try self.popBlock(
        alloc,
        .{ .ret = ret },
        null,
        root_name_hint,
    );
}

pub fn dump(
    self: *@This(),
) void {
    std.debug.print("IRGEN DUMP:\n", .{});
    for (self.blocks.items, 0..) |block, id| {
        std.debug.print(
            \\{f} = block ({s}):
            \\
        , .{
            BlockId{ .i = @intCast(id) },
            block.debug_name,
        });
        self.dumpBlock(block);
    }
    for (self.structs.items, 0..) |str, id| {
        std.debug.print(
            \\{f} = struct ({s}):
            \\    decl_block {f}
            \\
        , .{
            StructId{ .i = @intCast(id) },
            str.debug_name,
            str.decl_block,
        });
    }
    for (self.protos.items, 0..) |proto, id| {
        std.debug.print(
            \\{f} = proto ({s}):
            \\    is_extern {}
            \\    is_va_args {}
            \\    decl_block {f}
            \\
        , .{
            ProtoId{ .i = @intCast(id) },
            proto.debug_name,
            proto.is_extern,
            proto.is_va_args,
            proto.decl_block,
        });
    }
    for (self.functions.items, 0..) |func, id| {
        std.debug.print(
            \\{f} = function ({s}):
            \\    proto {f}
            \\    entry_block {f}
            \\
        , .{
            FunctionId{ .i = @intCast(id) },
            func.debug_name,
            func.proto,
            func.entry_block,
        });
    }
}

fn dumpBlock(
    self: *@This(),
    block: Block,
) void {
    for (block.instructions.start.i..block.instructions.end.i) |instr_i| {
        dumpInstr(self.instrs.items[instr_i]);
    }
    switch (block.branch_instruction) {
        .conditional => |v| {
            std.debug.print("    if {f} {f} else {f}\n", .{ v.boolean, v.on_true, v.on_false });
        },
        .unconditional => |v| {
            std.debug.print("    jump {f}\n", .{v});
        },
        .ret => |reg| {
            std.debug.print("    return {f}\n", .{reg});
        },
        .end => {
            std.debug.print("    decl_fn_end\n", .{});
        },
    }
}

fn dumpInstr(
    instr: Instr,
) void {
    switch (instr) {
        .str_lit => |v| {
            std.debug.print("    {f} = {s}\n", .{ v.result, v.value });
        },
        .int_lit => |v| {
            std.debug.print("    {f} = {}\n", .{ v.result, v.value });
        },
        .float_lit => |v| {
            std.debug.print("    {f} = {}\n", .{ v.result, v.value });
        },
        .void_lit => |v| {
            std.debug.print("    {f} = {{}}\n", .{v.result});
        },
        .builtin_lit => |v| {
            std.debug.print("    {f} = {t}\n", .{ v.result, v.builtin });
        },
        .call => |v| {
            std.debug.print("    {f} = call {f}\n", .{ v.result, v.func });
        },
        .unary_op => |v| {
            std.debug.print("    {f} = {f} {f}\n", .{ v.result, v.op, v.value });
        },
        .binary_op => |v| {
            std.debug.print("    {f} = {f} {f} {f}\n", .{ v.result, v.lhs, v.op, v.rhs });
        },
        .arg_fetch => |v| {
            std.debug.print("    {f} = arg_fetch\n", .{v.result});
        },
        .assign => |v| {
            std.debug.print("    {f} = {f}\n", .{ v.target, v.value });
        },
        .decl_fn => |v| {
            std.debug.print("    {f} = decl_fn {f}\n", .{ v.result, v.func });
        },
        .decl_proto => |v| {
            std.debug.print("    {f} = decl_proto {f}\n", .{ v.result, v.proto });
        },
        .decl_param => |v| {
            std.debug.print("    decl_param {f}\n", .{v.ty});
        },
        .decl_return => |v| {
            std.debug.print("    decl_return {f}\n", .{v.ty});
        },
        .decl_arg => |v| {
            std.debug.print("    decl_arg {f}\n", .{v.value});
        },
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

    if (decl.type_hint) |type_hint| {
        _ = type_hint;
        @panic("todo");
        // const type_hint = try self.convertExpr(alloc, &.{
        //     .prev = name_hint,
        //     .part = decl.ident.read(self.parser.tokenizer.source),
        // }, type_hint);
    }

    const result = try self.convertExpr(
        alloc,
        &next_name_hint,
        decl.expr,
    );

    try self.registers.renameReg(
        alloc,
        result,
        name,
    );
}

pub fn convertExpr(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
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
            lit.tok.read(self.source()),
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

    try self.builder.pushInstr(alloc, .{ .assign = .{
        .target = target,
        .value = value,
    } });
}

pub fn convertIf(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    const @"if" = self.nodes()[node_id].@"if";

    const name_hint_check = name_hint.push("check");
    const name_hint_if_entry = name_hint.push("if_entry");
    const name_hint_on_true = name_hint.push("on_true");
    const name_hint_on_false = name_hint.push("on_false");

    const boolean = try self.convertExpr(
        alloc,
        &name_hint_check,
        @"if".check_expr,
    );
    const result = self.registers.pushTmp();

    const if_entry_block = self.current_block;
    const continue_block = try self.allocBlock(alloc);

    log.info("if entry: {f}", .{if_entry_block});
    log.info("continue: {f}", .{continue_block});

    const on_true_block = try self.pushBlock(alloc);
    log.info("on true: {f}", .{on_true_block});
    const on_true_value = try self.convertScope(
        alloc,
        name_hint,
        @"if".on_true_scope,
    );
    try self.builder.pushInstr(alloc, .{ .assign = .{
        .target = result,
        .value = on_true_value,
    } });
    try self.popBlock(
        alloc,
        .{ .unconditional = continue_block },
        null,
        name_hint_on_true,
    );

    const on_false_block = try self.pushBlock(alloc);
    log.info("on false: {f}", .{on_false_block});
    const on_false_value = try self.convertScope(
        alloc,
        name_hint,
        @"if".on_false_scope,
    );
    try self.builder.pushInstr(alloc, .{ .assign = .{
        .target = result,
        .value = on_false_value,
    } });
    try self.popBlock(
        alloc,
        .{ .unconditional = continue_block },
        null,
        name_hint_on_false,
    );

    try self.popBlock(
        alloc,
        .{ .conditional = .{
            .boolean = boolean,
            .on_true = on_true_block,
            .on_false = on_false_block,
        } },
        if_entry_block,
        name_hint_if_entry,
    );

    try self.builder.pushBlock(alloc);
    self.current_block = continue_block;

    return result;
}

pub fn convertProtoId(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!ProtoId {
    const proto = self.nodes()[node_id].proto;

    const proto_id = try self.allocProto(alloc);

    const prev_block = self.current_block;
    const decl_block = try self.pushBlock(alloc);
    try self.pushScope(alloc);

    try self.param_stack.ensureUnusedCapacity(alloc, proto.params.len());
    const param_type_regs = self.param_stack.addManyAsSliceAssumeCapacity(proto.params.len());

    const name_hint_proto = name_hint.push("proto");
    const name_hint_param = name_hint_proto.push("param");
    for (proto.params.start..proto.params.end, param_type_regs) |param_node_id, *param_type_reg| {
        const param = self.nodes()[param_node_id].param;
        const param_name = param.ident.read(self.source());
        param_type_reg.* = try self.convertExpr(
            alloc,
            &name_hint_param.push(param_name),
            param.type,
        );
    }

    const name_hint_ret = name_hint_proto.push("ret");
    const return_type = if (proto.return_ty_expr) |expr_node_id|
        try self.convertExpr(
            alloc,
            &name_hint_ret,
            expr_node_id,
        )
    else
        try self.convertAccess(alloc, "void");

    for (param_type_regs) |param_type| {
        try self.builder.pushInstr(alloc, .{
            .decl_param = .{ .ty = param_type },
        });
    }
    try self.builder.pushInstr(alloc, .{
        .decl_return = .{ .ty = return_type },
    });

    self.popScope();
    try self.popBlock(
        alloc,
        .end,
        null,
        name_hint_proto,
    );
    self.current_block = prev_block;

    self.protos.items[proto_id.i] = Proto{
        .debug_name = try self.allocDebugName(name_hint.*),
        .decl_block = decl_block,
        .is_extern = proto.@"extern",
        .is_va_args = proto.is_va_args,
    };

    return proto_id;
}

pub fn convertFnId(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!FunctionId {
    const func = self.nodes()[node_id].@"fn";
    const proto = self.nodes()[func.proto].proto;
    const proto_id = try self.convertProtoId(alloc, name_hint, func.proto);

    const fn_id = try self.allocFunction(alloc);

    const prev_block = self.current_block;
    const entry_block = try self.pushBlock(alloc);
    try self.pushScope(alloc);

    const name_hint_fn = name_hint.push(if (proto.@"extern") "symexpr" else "fn");
    var return_value: RegId = undefined;
    if (proto.@"extern") {
        return_value = try self.convertExpr(
            alloc,
            &name_hint_fn,
            func.scope_or_symexpr,
        );
    } else {
        for (proto.params.start..proto.params.end) |param_node_id| {
            const param = self.nodes()[param_node_id].param;
            const param_name = param.ident.read(self.source());

            const result = self.registers.pushTmp();
            try self.registers.renameReg(alloc, result, param_name);
            try self.builder.pushInstr(alloc, .{ .arg_fetch = .{
                .result = result,
            } });
        }

        return_value = try self.convertScope(
            alloc,
            &name_hint_fn,
            func.scope_or_symexpr,
        );
    }

    self.popScope();
    try self.popBlock(
        alloc,
        .{ .ret = return_value },
        null,
        name_hint_fn,
    );
    self.current_block = prev_block;

    self.functions.items[fn_id.i] = Function{
        .debug_name = try self.allocDebugName(name_hint.*),
        .proto = proto_id,
        .entry_block = entry_block,
        // .parent = ,
    };

    return fn_id;
}

pub fn convertProto(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    const result = self.registers.pushTmp();
    const proto_id = try self.convertProtoId(
        alloc,
        name_hint,
        node_id,
    );

    try self.builder.pushInstr(alloc, .{ .decl_proto = .{
        .result = result,
        .proto = proto_id,
    } });
    return result;
}

pub fn convertFn(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    const result = self.registers.pushTmp();
    const func_id = try self.convertFnId(
        alloc,
        name_hint,
        node_id,
    );

    try self.builder.pushInstr(alloc, .{ .decl_fn = .{
        .result = result,
        .func = func_id,
    } });
    return result;
}

pub fn convertScope(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    const scope = self.nodes()[node_id].scope;

    const stmts, const last_stmt = scope.stmts.splitLast() orelse {
        return try self.convertVoidLit(alloc);
    };

    try self.pushScope(alloc);
    defer self.popScope();

    for (stmts.start..stmts.end) |stmt| {
        _ = try self.convertStmt(
            alloc,
            name_hint,
            @intCast(stmt),
        );
    }

    if (scope.has_trailing_semi) {
        _ = try self.convertStmt(
            alloc,
            name_hint,
            last_stmt,
        );
        return try self.convertVoidLit(alloc);
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

pub fn convertSlice(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    const slice = self.nodes()[node_id].slice;

    const elements = try self.convertExpr(
        alloc,
        name_hint,
        slice.elements_expr,
    );

    const result = self.registers.pushTmp();
    try self.builder.pushInstr(alloc, .{ .unary_op = .{
        .result = result,
        .value = elements,
        .op = if (slice.mut) .slice_mut else .slice,
    } });
    return result;
}

pub fn convertPointer(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    const pointer = self.nodes()[node_id].pointer;

    const elements = try self.convertExpr(
        alloc,
        name_hint,
        pointer.pointee_expr,
    );

    const result = self.registers.pushTmp();
    try self.builder.pushInstr(alloc, .{ .unary_op = .{
        .result = result,
        .value = elements,
        .op = if (pointer.mut) .pointer_mut else .pointer,
    } });
    return result;
}

pub fn convertBinaryOp(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
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

    const result = self.registers.pushTmp();
    try self.builder.pushInstr(alloc, .{ .binary_op = .{
        .result = result,
        .lhs = lhs,
        .rhs = rhs,
        .op = binary_op.op,
    } });
    return result;
}

pub fn convertFieldAcc(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    const field_acc = self.nodes()[node_id].field_acc;

    const container = try self.convertExpr(
        alloc,
        name_hint,
        field_acc.val,
    );

    const field = try self.convertStrLit(
        alloc,
        field_acc.ident.read(self.source()),
    );

    const result = self.registers.pushTmp();
    try self.builder.pushInstr(alloc, .{ .binary_op = .{
        .result = result,
        .lhs = container,
        .rhs = field,
        .op = BinaryOp.field,
    } });
    return result;
}

pub fn convertCall(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!RegId {
    const call = self.nodes()[node_id].call;

    for (call.args.start..call.args.end) |expr_node_id| {
        const arg = try self.convertExpr(
            alloc,
            name_hint,
            @intCast(expr_node_id),
        );

        try self.builder.pushInstr(alloc, .{ .decl_arg = .{
            .value = arg,
        } });
    }

    const func = try self.convertExpr(
        alloc,
        name_hint,
        call.val,
    );

    const result = self.registers.pushTmp();
    try self.builder.pushInstr(alloc, .{ .call = .{
        .result = result,
        .func = func,
    } });
    return result;
}

pub fn convertAccess(
    self: *@This(),
    alloc: std.mem.Allocator,
    var_name: []const u8,
) Error!RegId {
    // if (std.mem.eql(u8, "_", var_name)) {
    //     const result = self.registers.pushTmp();
    //     //
    //     return result;
    // }

    if (std.meta.stringToEnum(BuiltinVariable, var_name)) |builtin| {
        const result = self.registers.pushTmp();
        try self.builder.pushInstr(alloc, .{ .builtin_lit = .{
            .result = result,
            .builtin = builtin,
        } });
        return result;
    }

    const result = self.registers.findVar(var_name) orelse {
        log.debug("variable not found: {s}", .{var_name});
        return Error.VariableNotFound;
    };
    return result;
}

pub fn convertStrLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    value: []const u8,
) Error!RegId {
    const result = self.registers.pushTmp();
    try self.builder.pushInstr(alloc, .{ .str_lit = .{
        .result = result,
        .value = value,
    } });
    return result;
}

pub fn convertFloatLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    value: f64,
) Error!RegId {
    const result = self.registers.pushTmp();
    try self.builder.pushInstr(alloc, .{ .float_lit = .{
        .result = result,
        .value = value,
    } });
    return result;
}

pub fn convertIntLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    value: u128,
) Error!RegId {
    const result = self.registers.pushTmp();
    try self.builder.pushInstr(alloc, .{ .int_lit = .{
        .result = result,
        .value = value,
    } });
    return result;
}

pub fn convertVoidLit(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!RegId {
    const result = self.registers.pushTmp();
    try self.builder.pushInstr(alloc, .{ .void_lit = .{
        .result = result,
    } });
    return result;
}

pub const Registers = struct {
    var_name_hashmap: std.StringHashMapUnmanaged(ShadowChainEntry) = .{},
    shadow_chain: std.ArrayList(ShadowChainEntry) = .{},
    reg_names: std.AutoHashMapUnmanaged(RegId, []const u8) = .{},

    scope_reg_counters: std.ArrayList(u32) = .{},
    top_scope_reg_counter: u32 = 0,

    largest_reg: RegId = .{ .i = 0 },

    const null_reg: RegId = .{0};

    const ShadowChainEntry = struct {
        prev: u32,
        this: RegId,
    };

    pub fn deinit(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) void {
        self.scope_reg_counters.deinit(alloc);

        self.reg_names.deinit(alloc);
        self.shadow_chain.deinit(alloc);
        self.var_name_hashmap.deinit(alloc);
    }

    pub fn pushTmp(self: *@This()) RegId {
        const reg: RegId = .{ .i = self.top_scope_reg_counter };
        self.top_scope_reg_counter += 1;
        self.largest_reg.i = @max(self.largest_reg.i, reg.i);
        return reg;
    }

    pub fn renameReg(
        self: *@This(),
        alloc: std.mem.Allocator,
        reg: RegId,
        var_name: []const u8,
    ) Error!void {
        try self.var_name_hashmap.ensureUnusedCapacity(alloc, 1);
        try self.shadow_chain.ensureUnusedCapacity(alloc, 1);
        try self.reg_names.ensureUnusedCapacity(alloc, 1);

        self.reg_names.putAssumeCapacity(reg, var_name);
        const lookup_entry = self.var_name_hashmap.getOrPutAssumeCapacity(var_name);
        if (lookup_entry.found_existing) {
            const prev: u32 = @intCast(self.shadow_chain.items.len);
            self.shadow_chain.appendAssumeCapacity(lookup_entry.value_ptr.*);
            lookup_entry.value_ptr.* = .{
                .prev = prev,
                .this = reg,
            };
        } else {
            lookup_entry.value_ptr.* = .{
                .prev = std.math.maxInt(u32),
                .this = reg,
            };
        }
    }

    pub fn findVar(
        self: *@This(),
        var_name: []const u8,
    ) ?RegId {
        const entry = self.var_name_hashmap.get(var_name) orelse return null;
        return entry.this;
    }

    pub fn popVar(
        self: *@This(),
        var_name: []const u8,
    ) void {
        const lookup_entry = self.var_name_hashmap.getPtr(var_name) orelse return;

        if (lookup_entry.prev != std.math.maxInt(u32)) {
            lookup_entry.* = self.shadow_chain.items[lookup_entry.prev];
            self.shadow_chain.items[lookup_entry.prev] = undefined;
        } else {
            std.debug.assert(self.var_name_hashmap.remove(var_name));
        }
    }

    pub fn pushScope(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) Error!void {
        try self.scope_reg_counters.append(alloc, self.top_scope_reg_counter);
    }

    pub fn popScope(
        self: *@This(),
    ) void {
        const new_reg_counter = self.scope_reg_counters.pop() orelse 0;
        const score_reg_count = self.top_scope_reg_counter - new_reg_counter;
        self.top_scope_reg_counter = new_reg_counter;

        for (new_reg_counter..new_reg_counter + score_reg_count) |reg_id| {
            const reg: RegId = .{ .i = @intCast(reg_id) };
            const reg_name = (self.reg_names.fetchRemove(reg) orelse continue).value;
            self.popVar(reg_name);
        }
    }
};

pub const Builder = struct {
    instrs: std.ArrayList(Instr) = .{},
    block_instr_counts: std.ArrayList(u32) = .{},
    top_block_instr_count: u32 = 0,

    pub fn deinit(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) void {
        self.block_instr_counts.deinit(alloc);
        self.instrs.deinit(alloc);
    }

    pub fn pushInstr(
        self: *@This(),
        alloc: std.mem.Allocator,
        instr: Instr,
    ) Error!void {
        try self.instrs.append(alloc, instr);
        self.top_block_instr_count += 1;
    }

    pub fn pushBlock(
        self: *@This(),
        alloc: std.mem.Allocator,
    ) Error!void {
        try self.block_instr_counts.append(alloc, self.top_block_instr_count);
        self.top_block_instr_count = 0;
    }

    pub fn popBlock(
        self: *@This(),
        alloc: std.mem.Allocator,
        instrs_output: *std.ArrayList(Instr),
        branch_instr: BranchInstr,
        debug_name: []const u8,
    ) Error!Block {
        const instrs = self.top_block_instr_count;
        self.top_block_instr_count = self.block_instr_counts.pop() orelse 0;

        const instr: InstrRange = .{
            .start = .{ .i = @intCast(instrs_output.items.len) },
            .end = .{ .i = @intCast(instrs_output.items.len + instrs) },
        };
        const instrs_dst = try instrs_output.addManyAsSlice(alloc, instrs);
        const instrs_src = self.instrs.items[self.instrs.items.len - instrs ..];

        @memcpy(instrs_dst, instrs_src);
        self.instrs.items.len -= instrs;

        return Block{
            .debug_name = debug_name,
            .instructions = instr,
            .branch_instruction = branch_instr,
        };
    }
};
