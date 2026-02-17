const std = @import("std");
const IrGenerator = @import("IrGenerator.zig");
// const Range = @import("main.zig").Range;
const RegId = IrGenerator.RegId;
const Block = IrGenerator.Block;
const BlockId = IrGenerator.BlockId;
const Instr = IrGenerator.Instr;
const BranchInstr = IrGenerator.BranchInstr;
const Proto = IrGenerator.Proto;
const Function = IrGenerator.Function;
const FunctionId = IrGenerator.FunctionId;

function_typeids_resolved: std.DynamicBitSetUnmanaged = .{},
function_typeids: []const TypeId = &.{},
proto_typeids_resolved: std.DynamicBitSetUnmanaged = .{},
proto_typeids: []const TypeId = &.{},

types: std.ArrayList(Type) = .{},
type_ids: std.HashMapUnmanaged(Type, TypeId, TypeHashMapContext, 80) = .{},

fn_decl_param_stack: std.ArrayList(TypeId) = .{},
fn_decl_return: TypeId = undefined,
call_stack: std.ArrayList(CallStackFrame) = .{},
registers: []Value = &.{},

common_types: std.EnumArray(BasicType, TypeId) = undefined,
common_string: TypeId = undefined,

ir_gen: *const IrGenerator,

pub const TypeId = packed struct { i: u32 };
pub const BasicType = enum {
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
};
pub const Type = union(enum) {
    basic: BasicType,

    array: struct {
        elements: TypeId,
        len: usize,
    },
    slice: struct {
        elements: TypeId,
        mut: bool,
    },
    pointer: struct {
        elements: TypeId,
        mut: bool,
    },

    function: struct {
        is_extern: bool,
        is_va_args: bool,
        params: []const TypeId,
        ret: TypeId,
    },

    fn eq(
        self: @This(),
        other: @This(),
    ) bool {
        if (@intFromEnum(self) != @intFromEnum(other))
            return false;

        switch (std.meta.activeTag(self)) {
            .function,
            => return self.function.is_va_args == other.function.is_va_args and
                self.function.ret.i == other.function.ret.i and
                std.mem.eql(TypeId, self.function.params, other.function.params),

            inline else => |v| {
                const lhs = @field(self, @tagName(v));
                const rhs = @field(other, @tagName(v));
                return std.meta.eql(lhs, rhs);
            },
        }
    }

    fn hash(
        self: @This(),
    ) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&@intFromEnum(self)));

        switch (self) {
            .function => |v| {
                hasher.update(std.mem.asBytes(&v.is_va_args));
                hasher.update(std.mem.asBytes(&v.ret));
                hasher.update(std.mem.sliceAsBytes(v.params));
            },
            else => {
                hasher.update(std.mem.asBytes(&self));
            },
        }

        return hasher.final();
    }
};

pub const TypeHashMapContext = struct {
    pub fn hash(self: @This(), s: Type) u64 {
        _ = self;
        return s.hash();
    }
    pub fn eql(self: @This(), a: Type, b: Type) bool {
        _ = self;
        return a.eq(b);
    }
};

pub const CallStackFrame = struct {
    instr: []const Instr,
    branch_instr: BranchInstr,
    return_value: RegId,
    block_id: BlockId,
    mode: union(enum) {
        eval,
        proto,
        function: FunctionId,
    } = .eval,
    is_extern: bool = false,
    is_va_args: bool = false,
};

pub const Value = union(enum) {
    boolean: bool,
    string: []const u8,
    int: u128,
    float: f64,
    type: TypeId,
    func: FunctionId,
    void,
    undef,
};

pub const Error = error{
    OutOfMemory,
    TypeMismatch,
    StackUnderflow,
    InvalidIr,
    InvalidOp,
};

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    for (self.types.items) |ty| if (ty == .function) {
        alloc.free(ty.function.params);
    };

    alloc.free(self.registers);
    self.call_stack.deinit(alloc);
    self.fn_decl_param_stack.deinit(alloc);
    self.type_ids.deinit(alloc);
    self.types.deinit(alloc);

    alloc.free(self.proto_typeids);
    self.proto_typeids_resolved.deinit(alloc);
    alloc.free(self.function_typeids);
    self.function_typeids_resolved.deinit(alloc);
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!void {
    const min_types: u32 = comptime @intCast(std.enums.values(BasicType).len + 16);

    self.function_typeids_resolved = try .initEmpty(alloc, self.functions().len);
    self.function_typeids = try alloc.alloc(TypeId, self.functions().len);
    self.proto_typeids_resolved = try .initEmpty(alloc, self.protos().len);
    self.proto_typeids = try alloc.alloc(TypeId, self.protos().len);

    self.registers = try alloc.alloc(Value, self.ir_gen.registers.largest_reg.i + 1);
    try self.types.ensureUnusedCapacity(alloc, min_types);
    try self.type_ids.ensureUnusedCapacity(alloc, min_types * 2);

    for (std.enums.values(BasicType)) |basic| {
        const basic_type_id: TypeId = .{ .i = @intCast(self.types.items.len) };
        const basic_type: Type = .{ .basic = basic };
        self.types.appendAssumeCapacity(basic_type);
        self.type_ids.putAssumeCapacity(basic_type, basic_type_id);
    }
    self.common_string = try self.typeId(alloc, .{ .slice = .{
        .elements = self.common_types.get(.u8),
        .mut = false,
    } });

    _ = try self.pushCallStack(alloc, .{ .i = 0 });

    errdefer {
        for (self.call_stack.items, 0..) |frame, i| {
            std.debug.print("call{}: {f}\n", .{
                i, frame.block_id,
            });
        }
    }

    while (self.lastCallStack()) |top| {
        if (popInstr(top)) |instr| {
            try self.evalInstr(alloc, instr);
        } else {
            try self.evalBranchInstr(alloc, top.branch_instr);
        }
    }
}

pub fn dump(
    self: *@This(),
) void {
    _ = self; // autofix
}

fn blocks(
    self: *const @This(),
) []const Block {
    return self.ir_gen.blocks.items;
}

fn instrs(
    self: *const @This(),
) []const Instr {
    return self.ir_gen.instrs.items;
}

fn protos(
    self: *const @This(),
) []const Proto {
    return self.ir_gen.protos.items;
}

fn functions(
    self: *const @This(),
) []const Function {
    return self.ir_gen.functions.items;
}

/// duplicates params instead of taking ownership
fn typeId(
    self: *@This(),
    alloc: std.mem.Allocator,
    _ty: Type,
) Error!TypeId {
    var ty = _ty;
    if (ty == .basic) {
        return self.common_types.get(ty.basic);
    }

    try self.types.ensureUnusedCapacity(alloc, 1);

    const entry = try self.type_ids.getOrPut(alloc, ty);
    if (!entry.found_existing) {
        if (ty == .function) {
            errdefer _ = self.type_ids.remove(ty);
            ty.function.params = try alloc.dupe(TypeId, entry.key_ptr.*.function.params);
        }

        const type_id: TypeId = .{ .i = @intCast(self.types.items.len) };
        self.types.appendAssumeCapacity(ty);
        entry.value_ptr.* = type_id;
    }

    return entry.value_ptr.*;
}

fn typeInfo(
    self: *@This(),
    type_id: TypeId,
) Type {
    return self.types.items[type_id.i];
}

fn lastCallStack(
    self: *@This(),
) ?*CallStackFrame {
    if (self.call_stack.items.len == 0) return null;
    return &self.call_stack.items[self.call_stack.items.len - 1];
}

fn pushCallStack(
    self: *@This(),
    alloc: std.mem.Allocator,
    block_id: BlockId,
) Error!*CallStackFrame {
    const block = self.blocks()[block_id.i];
    try self.call_stack.append(alloc, .{
        .instr = self.instrs()[block.instructions.start.i..block.instructions.end.i],
        .branch_instr = block.branch_instruction,
        .return_value = .{ .i = 0 },
        .block_id = block_id,
    });
    return self.lastCallStack().?;
}

fn popInstr(
    f: *CallStackFrame,
) ?Instr {
    if (f.instr.len == 0) return null;
    const instr = f.instr[0];
    f.instr = f.instr[1..];
    return instr;
}

fn readReg(
    self: *@This(),
    reg: RegId,
) Error!Value {
    if (reg.i >= self.registers.len) {
        @branchHint(.cold);
        return Error.InvalidIr;
    }
    return self.registers[reg.i];
}

fn writeReg(
    self: *@This(),
    reg: RegId,
    val: Value,
) Error!void {
    if (reg.i >= self.registers.len) {
        @branchHint(.cold);
        return Error.InvalidIr;
    }
    self.registers[reg.i] = val;
}

fn evalInstr(
    self: *@This(),
    alloc: std.mem.Allocator,
    instr: Instr,
) Error!void {
    switch (instr) {
        .str_lit => |v| {
            try self.writeReg(v.result, .{ .string = v.value });
        },
        .int_lit => |v| {
            try self.writeReg(v.result, .{ .int = v.value });
        },
        .float_lit => |v| {
            try self.writeReg(v.result, .{ .float = v.value });
        },
        .void_lit => |v| {
            try self.writeReg(v.result, .void);
        },

        .builtin_lit => |v| {
            const val: Value = switch (v.builtin) {
                .false => .{ .boolean = false },
                .true => .{ .boolean = true },

                inline else => |builtin| .{ .type = self.common_types.get(
                    @field(BasicType, @tagName(builtin)),
                ) },
            };
            try self.writeReg(v.result, val);
        },
        // builtin_lit: struct {
        //     result: RegId,
        //     builtin: BuiltinVariable,
        // },
        // call: struct {
        //     result: RegId,
        //     func: RegId,
        // },
        .unary_op => |v| {
            const new_value = switch (try self.readReg(v.value)) {
                .boolean => |val| switch (v.op) {
                    .not => Value{ .boolean = !val },
                    else => return Error.InvalidOp,
                },
                .string => return Error.InvalidOp,
                .int => |val| switch (v.op) {
                    // .neg => Value{ .int = -val },
                    .not => Value{ .int = ~val },
                    else => return Error.InvalidOp,
                },
                .float => |val| switch (v.op) {
                    .neg => Value{ .float = -val },
                    else => return Error.InvalidOp,
                },
                .func => return Error.InvalidOp,
                .type => |val| switch (v.op) {
                    .slice => Value{ .type = try self.typeId(alloc, .{ .slice = .{
                        .elements = val,
                        .mut = false,
                    } }) },
                    .slice_mut => Value{ .type = try self.typeId(alloc, .{ .slice = .{
                        .elements = val,
                        .mut = true,
                    } }) },
                    .pointer => Value{ .type = try self.typeId(alloc, .{ .pointer = .{
                        .elements = val,
                        .mut = false,
                    } }) },
                    .pointer_mut => Value{ .type = try self.typeId(alloc, .{ .pointer = .{
                        .elements = val,
                        .mut = true,
                    } }) },
                    else => return Error.InvalidOp,
                },
                .undef => return Error.InvalidOp,
                .void => return Error.InvalidOp,
            };
            try self.writeReg(v.result, new_value);
        },
        // unary_op: struct {
        //     result: RegId,
        //     value: RegId,
        //     op: UnaryOp,
        // },
        // binary_op: struct {
        //     result: RegId,
        //     lhs: RegId,
        //     rhs: RegId,
        //     op: BinaryOp,
        // },
        // arg_fetch: struct {
        //     result: RegId,
        // },
        // assign: struct {
        //     target: RegId,
        //     value: RegId,
        // },
        .decl_fn => |v| {
            if (self.function_typeids_resolved.isSet(v.func.i)) {
                // memoized
                const type_id = self.function_typeids[v.func.i];
                try self.writeReg(v.result, .{ .type = type_id });
            } else {
                self.lastCallStack().?.return_value = v.result;
                const function = self.functions()[v.func.i];
                const proto = self.protos()[function.proto.i];

                const next_stack = try self.pushCallStack(alloc, proto.decl_block);
                next_stack.return_value = v.result;
                next_stack.mode = .{ .function = v.func };
                next_stack.is_extern = proto.is_extern;
                next_stack.is_va_args = proto.is_va_args;
            }
        },
        .decl_proto => |v| {
            if (self.proto_typeids_resolved.isSet(v.proto.i)) {
                // memoized
                const type_id = self.proto_typeids[v.proto.i];
                try self.writeReg(v.result, .{ .type = type_id });
            } else {
                self.lastCallStack().?.return_value = v.result;
                const proto = self.protos()[v.proto.i];

                const next_stack = try self.pushCallStack(alloc, proto.decl_block);
                next_stack.return_value = v.result;
                next_stack.mode = .proto;
                next_stack.is_extern = proto.is_extern;
                next_stack.is_va_args = proto.is_va_args;
            }
        },
        // decl_fn: struct {
        //     result: RegId,
        //     func: FunctionId,
        // },
        .decl_param => |v| {
            const type_id_value = try self.readReg(v.ty);
            if (type_id_value != .type) {
                std.debug.print("{t} != .type", .{type_id_value});
                return Error.TypeMismatch;
            }

            try self.fn_decl_param_stack.append(alloc, type_id_value.type);
        },
        .decl_return => |v| {
            const type_id_value = try self.readReg(v.ty);
            if (type_id_value != .type) {
                std.debug.print("{t} != .type", .{type_id_value});
                return Error.TypeMismatch;
            }

            self.fn_decl_return = type_id_value.type;
        },
        // decl_return: struct {
        //     ty: RegId,
        // },
        // decl_arg: struct {
        //     value: RegId,
        // },
        else => std.debug.panic("TODO: {}", .{instr}),
    }
}

fn evalBranchInstr(
    self: *@This(),
    alloc: std.mem.Allocator,
    instr: BranchInstr,
) Error!void {
    switch (instr) {
        .conditional => |v| {
            const val = try self.readReg(v.boolean);
            if (val != .boolean) {
                return Error.TypeMismatch;
            }

            const block_id = if (val.boolean)
                v.on_true
            else
                v.on_false;

            _ = try self.pushCallStack(alloc, block_id);
        },
        .unconditional => |v| {
            _ = try self.pushCallStack(alloc, v);
        },
        .ret => |v| {
            const return_dst = self.call_stack.pop() orelse {
                return Error.StackUnderflow;
            };

            try self.writeReg(return_dst.return_value, try self.readReg(v));
        },
        .end => {
            const eval_stack = self.call_stack.pop() orelse {
                return Error.StackUnderflow;
            };

            if (eval_stack.mode == .eval) return;

            const fn_type: Type = .{ .function = .{
                .is_extern = eval_stack.is_extern,
                .is_va_args = eval_stack.is_va_args,
                .params = self.fn_decl_param_stack.items,
                .ret = self.fn_decl_return,
            } };
            const fn_type_id = try self.typeId(alloc, fn_type);
            self.fn_decl_param_stack.clearRetainingCapacity();

            switch (eval_stack.mode) {
                .proto => {
                    try self.writeReg(eval_stack.return_value, .{
                        .type = fn_type_id,
                    });
                },
                .function => |id| {
                    try self.writeReg(eval_stack.return_value, .{
                        .func = id,
                    });
                },
                .eval => unreachable,
            }
        },
    }
}
