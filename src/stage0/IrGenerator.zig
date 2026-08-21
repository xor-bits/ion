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

pub const Value = struct {
    val: RValue,
    /// l-values are locators, they are stored as
    /// pointers and are dereferenced automagically
    is_locator: bool = false,

    pub fn assumeLValue(
        self: Value,
    ) RValue {
        std.debug.assert(self.is_locator);
        return self.val;
    }

    pub fn assumeRValue(
        self: Value,
    ) RValue {
        std.debug.assert(!self.is_locator);
        return self.val;
    }

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        if (self.is_locator) {
            try writer.writeAll("*(");
        }
        try self.val.format(writer);
        if (self.is_locator) {
            try writer.writeAll(")");
        }
    }
};

pub const RValue = enum(u32) {
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

    pub const builtin_count = @intFromEnum(RValue.undefined) + 1;

    pub fn asIndex(self: @This()) ?Instr.Id {
        return switch (self) {
            _ => @enumFromInt(@intFromEnum(self) - builtin_count),
            else => null,
        };
    }

    pub fn rval(
        self: RValue,
    ) Value {
        return .{
            .val = self,
            .is_locator = false,
        };
    }

    pub fn lval(
        self: RValue,
    ) Value {
        return .{
            .val = self,
            .is_locator = true,
        };
    }

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        if (self.asIndex()) |instr| {
            try writer.print("{f}", .{instr});
        } else {
            try writer.print("@{t}", .{self});
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

    pub fn addNodeIds(
        extras: *std.ArrayList(u32),
        alloc: std.mem.Allocator,
        argc: usize,
    ) error{OutOfMemory}!struct { Extra, []NodeId } {
        std.debug.assert(size(u32) == size(NodeId));
        const extra: Extra = @enumFromInt(extras.items.len);
        const params = try extras.addManyAsSlice(alloc, argc);
        return .{ extra, @ptrCast(params) };
    }

    pub fn getNodeIds(
        extras: []const u32,
        argv: Extra,
        argc: usize,
    ) []const NodeId {
        return @ptrCast(extras[@intFromEnum(argv) + argc ..][0..argc]);
    }

    pub const Proto = extern struct {
        return_type: RValue,
        /// number of `Param` following this
        /// `Proto` in the `extras` array
        param_count: u32,
    };

    pub const Param = extern struct {
        val: RValue,
    };
};

pub const Instr = union(enum) {
    pub const Id = enum(u32) {
        start,
        _,

        pub fn asValue(
            self: @This(),
        ) RValue {
            return @enumFromInt(@intFromEnum(self) + RValue.builtin_count);
        }

        pub fn format(
            self: @This(),
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try writer.print("%{}", .{@intFromEnum(self)});
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
        func: RValue,
        argv: Extra,
        argc: u32,
    },
    unary_op: struct {
        value: RValue,
        op: UnaryOp,
    },
    binary_op: struct {
        lhs: RValue,
        rhs: RValue,
        op: BinaryOp,
    },
    array: struct {
        len: RValue,
        child: RValue,
    },
    typeof: struct {
        val: RValue,
    },
    alloca: struct {
        ty: RValue,
        init: RValue,
        mut: bool,
    },
    load: struct {
        pointer: RValue,
    },
    store: struct {
        pointer: RValue,
        val: RValue,
    },
    // /// creates a new function or a global
    // /// only usable in a struct
    // decl: struct {
    //     name: Span,
    //     block_end: Instr.Id,
    // },
    /// creates a new anonymous function
    func: struct {
        proto: RValue,
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
        val: RValue,
    },
    /// loops back to the start of a block in code
    @"continue": struct {
        block: Instr.Id,
    },
    /// tells which source line:col the next instructions are from
    dbg_loc: struct {
        line: u32,
        col: u32,
    },
    /// tells which source variable name the value is from
    dbg_name: struct {
        name: Span,
        val: RValue,
        mut: bool,
    },
    /// prints a value at compile time
    dbg_print: struct {
        val: RValue,
    },
    /// a block of instructions which can return with a value
    block: struct {
        block_end: Instr.Id,
    },
    conditional: struct {
        boolean: RValue,
        on_true_block_end: Instr.Id,
        on_false_block_end: Instr.Id,
    },
    // unconditional: struct {
    //     dst: Instr.Index,
    // },
};

pub const ErrorMsg = struct {
    span: Span,
    message: []const u8,

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try writer.print("{s}", .{self.message});
    }
};

pub const Diagnostic = @import("main.zig").Diagnostic(ErrorMsg);

pub const Error = error{
    InvalidSemantic,
    OutOfMemory,
};

function_context: std.ArrayList(Instr.Id) = .empty,
break_context: std.ArrayList(Instr.Id) = .empty,
continue_context: std.ArrayList(Instr.Id) = .empty,
instrs: std.MultiArrayList(Instr) = .empty,
extras: std.ArrayList(u32) = .empty,
instr_spans: std.ArrayList(Span) = .empty,
errors: std.ArrayList(ErrorMsg) = .empty,
symbols: Symbols = .{},
main: Value = RValue.undefined.lval(),
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
    for (self.errors.items) |err| {
        alloc.free(err.message);
    }
    self.errors.deinit(alloc);
    self.instr_spans.deinit(alloc);
    self.symbols.deinit(alloc);
    self.extras.deinit(alloc);
    self.instrs.deinit(alloc);
    self.continue_context.deinit(alloc);
    self.break_context.deinit(alloc);
    self.function_context.deinit(alloc);
}

pub fn printErrors(
    self: *const @This(),
) void {
    for (self.errors.items) |err| {
        const diag = Diagnostic{
            .kind = .err,
            .msg = err,
            .src = .fromSpan(err.span, self.source()),
        };
        std.debug.print("{f}\n", .{diag});
    }
}

pub fn pushError(
    self: *@This(),
    alloc: std.mem.Allocator,
    span: Span,
    comptime fmt: []const u8,
    args: anytype,
) Error {
    @branchHint(.cold);
    const message = try std.fmt.allocPrint(alloc, fmt, args);
    try self.errors.append(alloc, .{
        .message = message,
        .span = span,
    });
    return Error.InvalidSemantic;
}

fn nodes(
    self: *const @This(),
) []const Node {
    return self.parser.nodes.items;
}

fn spans(
    self: *const @This(),
) []const Span {
    return self.parser.node_spans.items;
}

fn source(
    self: *const @This(),
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

fn pushInstrStrLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    value: Span,
    span: Span,
) Error!Value {
    return try self.pushInstrGetValue(
        alloc,
        .{ .str_lit = .{
            .value = value,
        } },
        span,
    );
}

fn pushInstrIntLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    value: u64,
    span: Span,
) Error!Value {
    return try self.pushInstrGetValue(
        alloc,
        .{ .int_lit = .{
            .value = value,
        } },
        span,
    );
}

fn pushInstrFloatLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    value: f64,
    span: Span,
) Error!Value {
    return try self.pushInstrGetValue(
        alloc,
        .{ .float_lit = .{
            .value = value,
        } },
        span,
    );
}

fn pushInstrCall(
    self: *@This(),
    alloc: std.mem.Allocator,
    func: Value,
    argv: Extra,
    argc: u32,
    span: Span,
) Error!Value {
    const func_rval = try self.loadRValue(
        alloc,
        func,
        span,
    );
    return try self.pushInstrGetValue(
        alloc,
        .{ .call = .{
            .func = func_rval,
            .argv = argv,
            .argc = argc,
        } },
        span,
    );
}

fn pushInstrUnaryOp(
    self: *@This(),
    alloc: std.mem.Allocator,
    value: Value,
    op: UnaryOp,
    span: Span,
) Error!Value {
    const rval = try self.loadRValue(
        alloc,
        value,
        span,
    );
    return try self.pushInstrGetValue(
        alloc,
        .{ .unary_op = .{
            .value = rval,
            .op = op,
        } },
        span,
    );
}

fn pushInstrBinaryOp(
    self: *@This(),
    alloc: std.mem.Allocator,
    lhs: Value,
    rhs: Value,
    op: BinaryOp,
    span: Span,
) Error!Value {
    const lhs_rval = try self.loadRValue(
        alloc,
        lhs,
        span,
    );
    const rhs_rval = try self.loadRValue(
        alloc,
        rhs,
        span,
    );
    return try self.pushInstrGetValue(
        alloc,
        .{ .binary_op = .{
            .lhs = lhs_rval,
            .rhs = rhs_rval,
            .op = op,
        } },
        span,
    );
}

fn pushInstrArray(
    self: *@This(),
    alloc: std.mem.Allocator,
    len: Value,
    child: Value,
    span: Span,
) Error!Value {
    const len_rval = try self.loadRValue(
        alloc,
        len,
        span,
    );
    const child_rval = try self.loadRValue(
        alloc,
        child,
        span,
    );
    return try self.pushInstrGetValue(
        alloc,
        .{ .array = .{
            .len = len_rval,
            .child = child_rval,
        } },
        span,
    );
}

fn pushInstrTypeof(
    self: *@This(),
    alloc: std.mem.Allocator,
    val: Value,
    span: Span,
) Error!Value {
    const rval = try self.loadRValue(
        alloc,
        val,
        span,
    );
    return try self.pushInstrGetValue(
        alloc,
        .{ .typeof = .{
            .val = rval,
        } },
        span,
    );
}

fn pushInstrAlloca(
    self: *@This(),
    alloc: std.mem.Allocator,
    ty: Value,
    init: Value,
    mut: bool,
    span: Span,
) Error!Value {
    const ty_rval = try self.loadRValue(
        alloc,
        ty,
        span,
    );
    const init_rval = try self.loadRValue(
        alloc,
        init,
        span,
    );
    const alloca_ptr = try self.pushInstrGetRValue(
        alloc,
        .{ .alloca = .{
            .ty = ty_rval,
            .init = init_rval,
            .mut = mut,
        } },
        span,
    );
    return alloca_ptr.lval();
}

fn pushInstrLoad(
    self: *@This(),
    alloc: std.mem.Allocator,
    pointer: Value,
    span: Span,
) Error!Value {
    const pointer_rval = try self.loadRValue(
        alloc,
        pointer,
        span,
    );
    return try self.pushInstrGetValue(
        alloc,
        .{ .load = .{
            .pointer = pointer_rval,
        } },
        span,
    );
}

fn pushInstrStore(
    self: *@This(),
    alloc: std.mem.Allocator,
    lval: Value,
    val: Value,
    span: Span,
) Error!void {
    if (!lval.is_locator) {
        @branchHint(.cold);
        return self.pushError(
            alloc,
            span,
            "cannot assign to a temporary",
            .{},
        );
    }
    const val_rval = try self.loadRValue(
        alloc,
        val,
        span,
    );
    _ = try self.pushInstr(
        alloc,
        .{ .store = .{
            .pointer = lval.val,
            .val = val_rval,
        } },
        span,
    );
}

fn pushInstrProto(
    self: *@This(),
    alloc: std.mem.Allocator,
    extra: Extra,
    span: Span,
) Error!Value {
    return try self.pushInstrGetValue(
        alloc,
        .{ .proto = .{
            .extra = extra,
        } },
        span,
    );
}

fn pushInstrParam(
    self: *@This(),
    alloc: std.mem.Allocator,
    span: Span,
) Error!Value {
    return try self.pushInstrGetValue(
        alloc,
        .param,
        span,
    );
}

fn pushInstrBreak(
    self: *@This(),
    alloc: std.mem.Allocator,
    block: Instr.Id,
    val: Value,
    span: Span,
) Error!void {
    const rval = try self.loadRValue(
        alloc,
        val,
        span,
    );
    _ = try self.pushInstr(
        alloc,
        .{ .@"break" = .{
            .block = block,
            .val = rval,
        } },
        span,
    );
}

fn pushInstrContinue(
    self: *@This(),
    alloc: std.mem.Allocator,
    block: Instr.Id,
    span: Span,
) Error!void {
    _ = try self.pushInstr(
        alloc,
        .{ .@"continue" = .{
            .block = block,
        } },
        span,
    );
}

// fn pushInstrDbgLoc(
//     self: *@This(),
//     alloc: std.mem.Allocator,
//     line: u32,
//     col: u32,
//     span: Span,
// ) Error!Value {
//     return try self.pushInstrGetValue(
//         alloc,
//         .{ .dbg_loc = .{
//             .line = line,
//             .col = col,
//         } },
//         span,
//     );
// }

fn pushInstrDbgPrint(
    self: *@This(),
    alloc: std.mem.Allocator,
    val: Value,
    span: Span,
) Error!Value {
    const rval = try self.loadRValue(
        alloc,
        val,
        span,
    );
    return try self.pushInstrGetValue(
        alloc,
        .{ .dbg_print = .{
            .val = rval,
        } },
        span,
    );
}

fn pushInstrUninitialized(
    self: *@This(),
    alloc: std.mem.Allocator,
    span: Span,
) Error!Instr.Id {
    return try self.pushInstr(
        alloc,
        undefined,
        span,
    );
}

fn pushInstr(
    self: *@This(),
    alloc: std.mem.Allocator,
    instr: Instr,
    span: Span,
) Error!Instr.Id {
    const id = self.instrs.len;
    std.debug.assert(id == self.instr_spans.items.len);
    try self.instrs.append(alloc, instr);
    try self.instr_spans.append(alloc, span);
    return @enumFromInt(id);
}

fn pushInstrGetRValue(
    self: *@This(),
    alloc: std.mem.Allocator,
    instr: Instr,
    span: Span,
) Error!RValue {
    const instr_addr = try self.pushInstr(
        alloc,
        instr,
        span,
    );
    return instr_addr.asValue();
}

fn pushInstrGetValue(
    self: *@This(),
    alloc: std.mem.Allocator,
    instr: Instr,
    span: Span,
) Error!Value {
    const rval = try self.pushInstrGetRValue(
        alloc,
        instr,
        span,
    );
    return .{
        .val = rval,
        .is_locator = false,
    };
}

fn loadRValue(
    self: *@This(),
    alloc: std.mem.Allocator,
    val: Value,
    span: Span,
) Error!RValue {
    if (!val.is_locator) {
        return val.val;
    }
    return try self.pushInstrGetRValue(
        alloc,
        .{ .load = .{
            .pointer = val.val,
        } },
        span,
    );
}

fn overwriteInstr(
    self: *@This(),
    idx: Instr.Id,
    instr: Instr,
) void {
    self.instrs.set(
        @intFromEnum(idx),
        instr,
    );
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

    const root_name_hint: NameHint = .new("root");
    try self.convertStructContents(
        alloc,
        &root_name_hint,
        0,
    );

    self.main = self.symbols.findVar("main") orelse {
        return self.pushError(
            alloc,
            .{},
            "no main function",
            .{},
        );
    };

    _ = try self.pushInstrCall(
        alloc,
        self.main,
        @enumFromInt(0),
        0,
        .{},
    );
}

pub fn dump(
    self: *@This(),
) void {
    std.debug.print("IR GENERATOR DUMP:\n", .{});

    // for (0..self.instrs.len) |i| {
    //     std.debug.print("%{} = {t}\n", .{ i, self.instrs.get(i) });
    // }

    self.dumpBlock(.start, @enumFromInt(self.instrs.len), 0);
    std.debug.print(";; instr extra = {}\n", .{self.extras.items.len});
    std.debug.print(";; instr count = {}\n", .{self.instrs.len});
    std.debug.print(";; main = {f}\n", .{self.main});
}

fn dumpBlock(
    self: *@This(),
    start: Instr.Id,
    end: Instr.Id,
    indent: usize,
) void {
    var cur = start;

    // std.debug.print("{f}..{f}\n", .{ start, end });

    while (@intFromEnum(cur) < @intFromEnum(end)) {
        const instr = self.instrs.get(@intFromEnum(cur));
        for (0..indent) |_| std.debug.print("    ", .{});
        std.debug.print("{f} = ", .{cur});
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
                std.debug.print("unary_op(op={f}, value={f})\n", .{
                    v.op,
                    v.value,
                });
            },
            .binary_op => |v| {
                std.debug.print("binary_op(op={f}, lhs={f}, rhs={f})\n", .{
                    v.op, v.lhs, v.rhs,
                });
            },
            .typeof => |v| {
                std.debug.print("typeof(val={f})\n", .{
                    v.val,
                });
            },
            .array => |v| {
                std.debug.print("array(len={f}, child={f})\n", .{
                    v.len, v.child,
                });
            },
            .alloca => |v| {
                std.debug.print("alloca(type={f}, mut={}, init={f})\n", .{
                    v.ty,
                    v.mut,
                    v.init,
                });
            },
            .load => |v| {
                std.debug.print("load(pointer={f})\n", .{
                    v.pointer,
                });
            },
            .store => |v| {
                std.debug.print("store(pointer={f}, val={f})\n", .{
                    v.pointer,
                    v.val,
                });
            },
            // .decl => |v| {
            //     std.debug.print("decl(name=\"{s}\", block={{\n", .{
            //         v.name.read(self.source()),
            //     });
            //     self.dumpBlock(cur, v.block_end, indent + 1);
            //     for (0..indent) |_| std.debug.print("    ", .{});
            //     std.debug.print("}})\n", .{});
            //     cur = v.block_end;
            // },
            .func => |v| {
                std.debug.print("func(proto={f}, body={{\n", .{v.proto});
                self.dumpBlock(cur, v.body_block_end, indent + 1);
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
                std.debug.print("break(block={f}, value={f})\n", .{ v.block, v.val });
            },
            .@"continue" => |v| {
                std.debug.print("continue(block={f})\n", .{v.block});
            },
            .dbg_loc => |v| {
                std.debug.print("dbg_loc(line={}, col={})\n", .{ v.line, v.col });
            },
            .dbg_name => |v| {
                std.debug.print("dbg_name(name=\"{s}\", val={f}, mut={})\n", .{
                    v.name.read(self.source()),
                    v.val,
                    v.mut,
                });
            },
            .dbg_print => |v| {
                std.debug.print("dbg_print(val={f})\n", .{v.val});
            },
            .block => |v| {
                std.debug.print("block(body={{\n", .{});
                self.dumpBlock(cur, v.block_end, indent + 1);
                for (0..indent) |_| std.debug.print("    ", .{});
                std.debug.print("}})\n", .{});
                cur = v.block_end;
            },
            .conditional => |v| {
                std.debug.print("conditional(check={f}, on_true_block={{\n", .{v.boolean});
                self.dumpBlock(cur, v.on_true_block_end, indent + 1);
                for (0..indent) |_| std.debug.print("    ", .{});
                std.debug.print("}}, on_false_block={{\n", .{});
                self.dumpBlock(v.on_true_block_end, v.on_false_block_end, indent + 1);
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
    const span = self.spans()[node_id];

    var val = try self.convertExpr(
        alloc,
        &next_name_hint,
        decl.expr,
    );

    var ty: Value = undefined;
    if (decl.type_hint) |type_hint| {
        ty = try self.convertExpr(
            alloc,
            &init_name_hint,
            type_hint,
        );

        val = try self.pushInstrBinaryOp(
            alloc,
            val,
            ty,
            .as,
            self.spans()[type_hint],
        );
    } else {
        ty = try self.pushInstrTypeof(
            alloc,
            val,
            span,
        );
    }

    const alloca = try self.pushInstrAlloca(
        alloc,
        ty,
        val,
        decl.mut,
        span,
    );

    try self.symbols.createVar(
        alloc,
        name,
        alloca,
    );
}

pub fn convertExpr(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    switch (self.nodes()[node_id]) {
        .print => return try self.convertComptimePrint(
            alloc,
            name_hint,
            node_id,
        ),
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
        .loop => return try self.convertLoop(
            alloc,
            name_hint,
            node_id,
        ),
        .assign => return try self.convertAssign(
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
        .address => return try self.convertAddress(
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
        .index_acc => return try self.convertIndexAcc(
            alloc,
            name_hint,
            node_id,
        ),
        .call => return try self.convertCall(
            alloc,
            name_hint,
            node_id,
        ),
        .access => return try self.convertAccess(
            alloc,
            node_id,
        ),
        .str_lit => return try self.convertStrLit(
            alloc,
            node_id,
        ),
        .float_lit => return try self.convertFloatLit(
            alloc,
            node_id,
        ),
        .int_lit => return try self.convertIntLit(
            alloc,
            node_id,
        ),
        else => std.debug.panic("TODO: {}", .{self.nodes()[node_id]}),
    }
}

pub fn convertAssign(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const assign = self.nodes()[node_id].assign;

    const target = try self.convertExpr(
        alloc,
        name_hint,
        assign.lhs,
    );

    const val = try self.convertExpr(
        alloc,
        name_hint,
        assign.rhs,
    );

    try self.pushInstrStore(
        alloc,
        target,
        val,
        self.spans()[node_id],
    );

    return RValue.void.rval();
}

pub fn convertComptimePrint(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const comptime_print = self.nodes()[node_id].print;
    const val = try self.convertExpr(
        alloc,
        name_hint,
        comptime_print.expr,
    );
    const rval = try self.loadRValue(
        alloc,
        val,
        self.spans()[node_id],
    );
    _ = try self.pushInstr(
        alloc,
        .{ .dbg_print = .{ .val = rval } },
        self.spans()[node_id],
    );
    return RValue.void.rval();
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

    const if_block = try self.pushInstr(
        alloc,
        .{ .block = undefined },
        self.spans()[node_id],
    );
    const if_block_entry = self.nextInstr();

    const boolean = try self.convertExpr(
        alloc,
        &name_hint_check,
        @"if".check_expr,
    );
    const boolean_rval = try self.loadRValue(
        alloc,
        boolean,
        self.spans()[@"if".check_expr],
    );

    const conditional = try self.pushInstr(
        alloc,
        .{ .conditional = undefined },
        self.spans()[node_id],
    );

    const on_true_val = try self.convertScope(
        alloc,
        &name_hint_on_true,
        @"if".on_true_scope,
    );
    try self.pushInstrBreak(
        alloc,
        if_block_entry,
        on_true_val,
        self.spans()[@"if".on_true_scope],
    );
    const on_true_block_end = self.nextInstr();

    const on_false_val = try self.convertScope(
        alloc,
        &name_hint_on_false,
        @"if".on_false_scope,
    );
    try self.pushInstrBreak(
        alloc,
        if_block_entry,
        on_false_val,
        self.spans()[@"if".on_false_scope],
    );
    const on_false_block_end = self.nextInstr();

    self.instrs.set(@intFromEnum(conditional), .{ .conditional = .{
        .boolean = boolean_rval,
        .on_true_block_end = on_true_block_end,
        .on_false_block_end = on_false_block_end,
    } });
    self.instrs.set(@intFromEnum(if_block), .{ .block = .{
        .block_end = on_false_block_end,
    } });

    return if_block.asValue().rval();
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
        const param_type_rval = try self.loadRValue(
            alloc,
            param_type,
            self.spans()[param.type],
        );
        const param_value = try self.pushInstrBinaryOp(
            alloc,
            RValue.undefined.rval(),
            param_type_rval.rval(),
            .as,
            self.spans()[proto.params.start + i],
        );

        try self.symbols.createVar(alloc, param_name, param_value);
        params_extra[i] = .{ .val = param_type_rval };
    }

    const return_type_rval = if (proto.return_ty_expr) |expr_node_id| b: {
        const name_hint_ret = name_hint_proto.push("ret");
        const return_type = try self.convertExpr(
            alloc,
            &name_hint_ret,
            expr_node_id,
        );
        break :b try self.loadRValue(
            alloc,
            return_type,
            self.spans()[expr_node_id],
        );
    } else RValue.void_type;

    proto_extra.* = .{
        .param_count = proto.params.len(),
        .return_type = return_type_rval,
    };

    return try self.pushInstrProto(
        alloc,
        extra,
        self.spans()[node_id],
    );
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

    const func_block = try self.pushInstr(
        alloc,
        .{ .func = undefined },
        self.spans()[node_id],
    );
    const func_block_start = self.nextInstr();

    const name_hint_fn = name_hint.push(if (proto_node.@"extern") "symexpr" else "fn");
    if (proto_node.@"extern") {
        const symbol = try self.convertExpr(
            alloc,
            &name_hint_fn,
            func_node.scope_or_symexpr,
        );
        try self.pushInstrBreak(
            alloc,
            func_block_start,
            symbol,
            self.spans()[func_node.scope_or_symexpr],
        );
    } else {
        const argc = proto_node.params.len();

        for (0..argc) |i| {
            const arg = try self.pushInstrParam(
                alloc,
                self.spans()[proto_node.params.start + i],
            );
            const val: Instr.Id = @enumFromInt(@intFromEnum(func_block_start) + i);
            std.debug.assert(arg.val.asIndex() == val);
        }

        for (0..argc) |i| {
            const param = self.nodes()[proto_node.params.start + i].param;
            const param_instr: Instr.Id = @enumFromInt(@intFromEnum(func_block_start) + i);
            const param_val = param_instr.asValue().rval();
            const span = self.spans()[proto_node.params.start + i];

            // FIXME: inefficient: building the function proto
            // type already creates values for param types
            const param_type = try self.pushInstrTypeof(
                alloc,
                param_val,
                span,
            );

            const param_alloca = try self.pushInstrAlloca(
                alloc,
                param_type,
                param_val,
                false,
                span,
            );
            try self.symbols.createVar(
                alloc,
                param.ident.read(self.source()),
                param_alloca,
            );
        }

        try self.function_context.append(
            alloc,
            func_block_start,
        );
        defer _ = self.function_context.pop();

        const return_value = try self.convertScope(
            alloc,
            &name_hint_fn,
            func_node.scope_or_symexpr,
        );
        try self.pushInstrBreak(
            alloc,
            func_block_start,
            return_value,
            self.spans()[node_id],
        );
    }
    const body_block_end = self.nextInstr();

    self.overwriteInstr(func_block, .{ .func = .{
        .proto = proto.assumeRValue(),
        .body_block_end = body_block_end,
    } });
    return func_block.asValue().rval();
}

pub fn convertLoop(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const loop = self.nodes()[node_id].loop;

    const loop_block = try self.pushInstr(
        alloc,
        .{ .block = undefined },
        self.spans()[node_id],
    );
    const loop_entry = self.nextInstr();

    try self.break_context.append(alloc, loop_entry);
    defer _ = self.break_context.pop();
    try self.continue_context.append(alloc, loop_entry);
    defer _ = self.break_context.pop();

    // TODO: give a warning when the loop scope tries to break a
    // value implicitly, because loops have an implicit continue
    // statement at the end
    _ = try self.convertScope(
        alloc,
        name_hint,
        loop.scope,
    );

    try self.pushInstrContinue(
        alloc,
        loop_entry,
        self.spans()[node_id],
    );

    const loop_block_end = self.nextInstr();
    self.overwriteInstr(loop_block, .{ .block = .{
        .block_end = loop_block_end,
    } });

    return RValue.void.rval();
}

pub fn convertScope(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const scope = self.nodes()[node_id].scope;

    const stmts, const last_stmt = scope.stmts.splitLast() orelse {
        return RValue.void.rval();
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
        return RValue.void.rval();
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
        .@"return" => try self.convertReturn(
            alloc,
            name_hint,
            node_id,
        ),
        .@"break" => try self.convertBreak(
            alloc,
            name_hint,
            node_id,
        ),
        .@"continue" => try self.convertContinue(
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

pub fn convertReturn(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const ret = self.nodes()[node_id].@"return";

    const ret_val = if (ret.value) |val|
        try self.convertExpr(
            alloc,
            name_hint,
            val,
        )
    else
        RValue.void.rval();

    const func = self.function_context.getLastOrNull() orelse {
        return self.pushError(
            alloc,
            self.spans()[node_id],
            "cannot return outside of a function",
            .{},
        );
    };
    try self.pushInstrBreak(
        alloc,
        func,
        ret_val,
        self.spans()[node_id],
    );
}

pub fn convertBreak(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const br = self.nodes()[node_id].@"break";

    const br_val = if (br.value) |val|
        try self.convertExpr(
            alloc,
            name_hint,
            val,
        )
    else
        RValue.void.rval();

    const block = self.break_context.getLastOrNull() orelse {
        return self.pushError(
            alloc,
            self.spans()[node_id],
            "cannot continue outside of a loop",
            .{},
        );
    };
    try self.pushInstrBreak(
        alloc,
        block,
        br_val,
        self.spans()[node_id],
    );
}

pub fn convertContinue(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    _ = name_hint;
    // const cont = self.nodes()[node_id].@"continue";

    const block = self.continue_context.getLastOrNull() orelse {
        return self.pushError(
            alloc,
            self.spans()[node_id],
            "cannot continue outside of a loop",
            .{},
        );
    };
    _ = try self.pushInstr(
        alloc,
        .{ .@"continue" = .{
            .block = block,
        } },
        self.spans()[node_id],
    );
}

pub fn convertArray(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const array = self.nodes()[node_id].array;

    const len = try self.convertExpr(
        alloc,
        &name_hint.push("length"),
        array.length_expr,
    );
    const child = try self.convertExpr(
        alloc,
        &name_hint.push("element"),
        array.elements_expr,
    );
    return try self.pushInstrArray(
        alloc,
        len,
        child,
        self.spans()[node_id],
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
    return try self.pushInstrUnaryOp(
        alloc,
        elements,
        if (slice.mut) .slice_mut else .slice,
        self.spans()[node_id],
    );
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
    return try self.pushInstrUnaryOp(
        alloc,
        elements,
        if (pointer.mut) .pointer_mut else .pointer,
        self.spans()[node_id],
    );
}

pub fn convertAddress(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const address = self.nodes()[node_id].address;

    const pointee = try self.convertExpr(
        alloc,
        name_hint,
        address.pointee_expr,
    );

    // self.pushInstr(alloc, .{
    //     .alloca = pointee,
    // }, self.spans()[node_id]);

    _ = pointee;
    @panic("todo");
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

    return try self.pushInstrBinaryOp(
        alloc,
        lhs,
        rhs,
        binary_op.op,
        self.spans()[node_id],
    );
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

    const field = try self.pushInstrGetValue(
        alloc,
        .{ .str_lit = .{ .value = field_acc.ident } },
        self.spans()[node_id],
    );

    return try self.pushInstrBinaryOp(
        alloc,
        container,
        field,
        .field,
        self.spans()[node_id],
    );
}

pub fn convertIndexAcc(
    self: *@This(),
    alloc: std.mem.Allocator,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!Value {
    const index_acc = self.nodes()[node_id].index_acc;

    const container = try self.convertExpr(
        alloc,
        name_hint,
        index_acc.val,
    );
    const index = try self.convertExpr(
        alloc,
        name_hint,
        index_acc.expr,
    );

    return try self.pushInstrBinaryOp(
        alloc,
        container,
        index,
        .index,
        self.spans()[node_id],
    );
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
    const arg_node_ids_extra, const arg_node_ids = try Extra.addNodeIds(
        &self.extras,
        alloc,
        argc,
    );
    std.debug.assert(@intFromEnum(arg_node_ids_extra) == @intFromEnum(argv) + argc);

    for (call.args.start..call.args.end, args, arg_node_ids) |expr_node_id, *arg, *arg_node_id| {
        const i: NodeId = @intCast(expr_node_id);
        const arg_expr_result = try self.convertExpr(
            alloc,
            name_hint,
            i,
        );
        const arg_expr_result_rval = try self.loadRValue(
            alloc,
            arg_expr_result,
            self.spans()[i],
        );
        arg.* = .{ .val = arg_expr_result_rval };
        arg_node_id.* = i;
    }

    const func = try self.convertExpr(
        alloc,
        name_hint,
        call.val,
    );

    return try self.pushInstrCall(
        alloc,
        func,
        argv,
        argc,
        self.spans()[call.val],
    );
}

pub fn convertAccess(
    self: *@This(),
    alloc: std.mem.Allocator,
    node_id: NodeId,
) Error!Value {
    const var_name = self.nodes()[node_id].access.ident.read(self.source());

    // if (std.mem.eql(u8, "_", var_name)) {
    //     const result = self.registers.pushTmp();
    //     //
    //     return result;
    // }

    if (std.meta.stringToEnum(BuiltinVariable, var_name)) |builtin| {
        return (switch (builtin) {
            .type => RValue.type_type,
            .void => RValue.void_type,
            .bool => RValue.bool_type,
            .u8 => RValue.u8_type,
            .u16 => RValue.u16_type,
            .u32 => RValue.u32_type,
            .u64 => RValue.u64_type,
            .usize => RValue.usize_type,
            .i8 => RValue.i8_type,
            .i16 => RValue.i16_type,
            .i32 => RValue.i32_type,
            .i64 => RValue.i64_type,
            .isize => RValue.isize_type,
            .f32 => RValue.f32_type,
            .f64 => RValue.f64_type,

            .c_int => RValue.c_int_type,
            .c_char => RValue.c_char_type,
            .c_long => RValue.c_long_type,
            .c_longlong => RValue.c_longlong_type,
            .c_short => RValue.c_short_type,
            .c_uint => RValue.c_uint_type,
            .c_ulong => RValue.c_ulong_type,
            .c_ulonglong => RValue.c_ulonglong_type,
            .c_ushort => RValue.c_ushort_type,

            .false => RValue.false,
            .true => RValue.true,
            .undefined => RValue.undefined,
        }).rval();
    }

    return self.symbols.findVar(var_name) orelse {
        return self.pushError(
            alloc,
            self.spans()[node_id],
            "variable not found",
            .{},
        );
    };
}

pub fn convertStrLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    node_id: NodeId,
) Error!Value {
    const span = self.nodes()[node_id].str_lit.tok;
    const contents = span.read(self.source());

    std.debug.assert(contents.len >= 2);
    std.debug.assert(contents[0] == '"');
    std.debug.assert(contents[contents.len - 1] == '"');

    var span_without_quotes = span;
    span_without_quotes.start += 1;
    span_without_quotes.end -= 1;

    return try self.pushInstrStrLit(
        alloc,
        span_without_quotes,
        self.spans()[node_id],
    );
}

pub fn convertFloatLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    node_id: NodeId,
) Error!Value {
    const value = self.nodes()[node_id].float_lit.val;
    return try self.pushInstrFloatLit(
        alloc,
        value,
        self.spans()[node_id],
    );
}

pub fn convertIntLit(
    self: *@This(),
    alloc: std.mem.Allocator,
    node_id: NodeId,
) Error!Value {
    // TODO: support big ints
    const value = self.nodes()[node_id].int_lit.val;
    return try self.pushInstrIntLit(
        alloc,
        value,
        self.spans()[node_id],
    );
}

pub const Symbols = struct {
    var_name_hashmap: std.StringHashMapUnmanaged(ShadowChainEntry) = .empty,
    shadow_chain: std.ArrayList(ShadowChainEntry) = .empty,
    val_names: std.AutoHashMapUnmanaged(RValue, []const u8) = .empty,

    /// holds the number of values in the value stack at that scopes position
    scope_sizes: std.ArrayList(u32) = .empty,
    /// holds all values of all scopes
    value_stack: std.ArrayList(RValue) = .empty,

    const ShadowChainEntry = struct {
        prev: u32,
        this: RValue,
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
        std.debug.assert(val.is_locator);

        try self.var_name_hashmap.ensureUnusedCapacity(alloc, 1);
        try self.shadow_chain.ensureUnusedCapacity(alloc, 1);
        try self.val_names.ensureUnusedCapacity(alloc, 1);
        try self.value_stack.ensureUnusedCapacity(alloc, 1);
        std.debug.assert(self.scope_sizes.items.len != 0);

        self.scope_sizes.items[self.scope_sizes.items.len - 1] += 1;
        self.value_stack.appendAssumeCapacity(val.val);

        self.val_names.putAssumeCapacity(val.val, var_name);
        const lookup_entry = self.var_name_hashmap.getOrPutAssumeCapacity(var_name);
        if (lookup_entry.found_existing) {
            const prev: u32 = @intCast(self.shadow_chain.items.len);
            self.shadow_chain.appendAssumeCapacity(lookup_entry.value_ptr.*);
            lookup_entry.value_ptr.* = .{
                .prev = prev,
                .this = val.val,
            };
        } else {
            lookup_entry.value_ptr.* = .{
                .prev = std.math.maxInt(u32),
                .this = val.val,
            };
        }
    }

    pub fn findVar(
        self: *@This(),
        var_name: []const u8,
    ) ?Value {
        const entry = self.var_name_hashmap.get(var_name) orelse return null;
        return entry.this.lval();
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
