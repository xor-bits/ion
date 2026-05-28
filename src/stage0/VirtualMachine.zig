const std = @import("std");

const ops = @import("VirtualMachine/ops.zig");

const IrGenerator = @import("IrGenerator.zig");
const Instr = IrGenerator.Instr;
const Value = IrGenerator.Value;
const Span = @import("Tokenizer.zig").Span;
const VirtualMachine = @This();

pub const EvalMode = enum {
    /// evaluate everything, including loops
    /// ; comptime
    eval,
    /// generate code and evaluate loops only once
    /// ; runtime
    compile,
};

pub const Frame = struct {
    block: Instr.Id = .start,
    instr: Instr.Id = .start,
    function: ?Instr.Id = null,
    registers: std.AutoHashMapUnmanaged(Instr.Id, Register) = .{},
    arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator),
    node: std.SinglyLinkedList.Node = .{},
    return_register: ?Instr.Id = null,
    return_type: ?Type.Id = null,
    symbol: Span = .{},
    // mode: EvalMode = .compile,

    pub fn clear(
        self: *@This(),
    ) void {
        self.instr = self.block;
        _ = self.arena.reset(.retain_capacity);
        self.registers.clearRetainingCapacity();
        self.return_register = null;
        self.return_type = null;
        self.symbol = .{};
    }
};

pub const Register = struct {
    type: Type.Id,
    val: PrimitiveValue,
    name: Span = .{},

    fn ty(
        type_id: Type.Id,
    ) @This() {
        return .{ .type = .type, .val = .{
            .type = type_id,
        } };
    }

    fn voidLit() @This() {
        return .{ .type = .void, .val = .void };
    }

    fn boolLit(
        b: bool,
    ) @This() {
        return .{ .type = .bool, .val = .{
            .bool = b,
        } };
    }

    fn undefinedLit() @This() {
        return .{ .type = .undefined, .val = .undefined };
    }

    fn strLit(
        s: []const u8,
    ) @This() {
        return .{ .type = .slice_u8, .val = .{
            .string = s,
        } };
    }

    fn intLit(
        i: u64,
    ) @This() {
        switch (Type.Id._isize) {
            .i64 => return .{ .type = ._isize, .val = .{
                .i64 = @bitCast(i),
            } },
            .i32 => return .{ .type = ._isize, .val = .{
                .i32 = @bitCast(i),
            } },
            else => comptime @compileError("unsupported host arch"),
        }
    }

    fn floatLit(
        f: f64,
    ) @This() {
        return .{ .type = .f64, .val = .{
            .f64 = f,
        } };
    }

    fn func(
        proto: Type.Id,
        entry: Instr.Id,
    ) @This() {
        return .{ .type = proto, .val = .{
            .func = .{ .entry = entry },
        } };
    }

    fn runtime(t: Type.Id) @This() {
        return .{ .type = t, .val = .runtime };
    }
};

const Signedness = enum {
    signed,
    unsigned,

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try writer.writeByte(switch (self) {
            .signed => 'i',
            .unsigned => 'u',
        });
    }
};

const IntSize = enum {
    @"8",
    @"16",
    @"32",
    @"64",

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try writer.writeAll(switch (self) {
            .@"8" => "8",
            .@"16" => "16",
            .@"32" => "32",
            .@"64" => "64",
        });
    }
};

const FloatSize = enum {
    @"32",
    @"64",

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try writer.writeAll(switch (self) {
            .@"32" => "32",
            .@"64" => "64",
        });
    }
};

pub const TypeInfo = struct {
    alignment: usize,
    size: usize,

    pub inline fn of(
        comptime T: type,
    ) @This() {
        return .{
            .alignment = @alignOf(T),
            .size = @sizeOf(T),
        };
    }

    pub fn ofInt(
        bits: IntSize,
    ) @This() {
        return switch (bits) {
            .@"8" => .{ .alignment = 1, .size = 1 },
            .@"16" => .{ .alignment = 2, .size = 2 },
            .@"32" => .{ .alignment = 4, .size = 4 },
            .@"64" => .{ .alignment = 8, .size = 8 },
        };
    }

    pub fn ofFloat(
        bits: FloatSize,
    ) @This() {
        return switch (bits) {
            .@"32" => .{ .alignment = 4, .size = 4 },
            .@"64" => .{ .alignment = 8, .size = 8 },
        };
    }

    pub fn pad(
        self: @This(),
    ) @This() {
        std.debug.assert(self.alignment <= self.size);
        return .{
            .alignment = self.alignment,
            .size = std.mem.alignForward(
                usize,
                self.size,
                self.alignment,
            ),
        };
    }

    pub fn repeat(
        self: @This(),
        n: usize,
    ) ?@This() {
        const padded = self.pad();
        return .{
            .alignment = padded.alignment,
            .size = std.math.mul(usize, padded.size, n) catch
                return null,
        };
    }
};

pub const Type = union(enum) {
    pub const Id = enum(u32) {
        /// common types
        type,
        void,
        bool,
        undefined,
        u8,
        u16,
        u32,
        u64,
        i8,
        i16,
        i32,
        i64,
        f32,
        f64,
        slice_u8,
        /// other types
        _,

        const _usize: Type.Id = baseU(usize);
        const _isize: Type.Id = baseI(isize);
        const _c_int: Type.Id = baseU(c_int);
        const _c_char: Type.Id = baseU(c_char);
        const _c_long: Type.Id = baseU(c_long);
        const _c_longlong: Type.Id = baseU(c_longlong);
        const _c_short: Type.Id = baseU(c_short);
        const _c_uint: Type.Id = baseU(c_uint);
        const _c_ulong: Type.Id = baseU(c_ulong);
        const _c_ulonglong: Type.Id = baseU(c_ulonglong);
        const _c_ushort: Type.Id = baseU(c_ushort);

        fn baseI(comptime T: type) Type.Id {
            return switch (@sizeOf(T)) {
                8 => .i64,
                4 => .i32,
                2 => .i16,
                1 => .i8,
                else => @compileError("unsupported architecture"),
            };
        }

        fn baseU(comptime T: type) Type.Id {
            return switch (@sizeOf(T)) {
                8 => .u64,
                4 => .u32,
                2 => .u16,
                1 => .u8,
                else => @compileError("unsupported architecture"),
            };
        }

        pub const Printer = struct {
            self: Type.Id,
            vm: *const VirtualMachine,

            pub fn format(
                self: @This(),
                writer: *std.Io.Writer,
            ) std.Io.Writer.Error!void {
                try self.self.write(self.vm, writer);
            }
        };

        fn write(
            self: @This(),
            vm: *const VirtualMachine,
            writer: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            const ty = vm.readType(self);
            switch (ty) {
                .int => |v| try writer.print("{f}{f}", .{ v.sign, v.bits }),
                .float => |v| try writer.print("f{f}", .{v.bits}),
                .array => |v| {
                    try writer.print("[{}]", .{v.len});
                    try v.child.write(vm, writer);
                },
                .slice => |v| {
                    try writer.writeAll("[]");
                    if (v.mut) try writer.writeAll("mut ");
                    try v.child.write(vm, writer);
                },
                .pointer => |v| {
                    try writer.writeAll("*");
                    if (v.mut) try writer.writeAll("mut ");
                    try v.child.write(vm, writer);
                },
                .func => |v| {
                    // FIXME: this is not the actual syntax,
                    // but func does not store the symexpr
                    if (v.@"extern") try writer.writeAll("extern ");
                    try writer.writeAll("fn(");
                    for (v.params) |t| {
                        try t.write(vm, writer);
                        try writer.writeByte(',');
                    }
                    try writer.writeAll(") ");
                    try v.@"return".write(vm, writer);
                },
                else => try writer.print("{t}", .{ty}),
            }
        }

        pub fn print(
            self: @This(),
            vm: *const VirtualMachine,
        ) Printer {
            return .{ .self = self, .vm = vm };
        }
    };

    type,
    void,
    bool,
    undefined,
    int: struct {
        sign: Signedness,
        bits: IntSize,
    },
    float: struct {
        bits: FloatSize,
    },
    array: struct {
        len: usize,
        child: Type.Id,
    },
    slice: struct {
        mut: bool,
        child: Type.Id,
    },
    pointer: struct {
        mut: bool,
        child: Type.Id,
    },
    // @"struct": struct {

    // },
    func: struct {
        @"extern": bool,
        /// this memory is managed `arena` when this Type is stored,
        /// but by anything (`alloc`) before resolve
        params: []Type.Id,
        @"return": Type.Id,
    },

    const Context = struct {
        pub fn hash(_: @This(), key: Type) u64 {
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHashStrat(&hasher, key, .DeepRecursive);
            return hasher.final();
        }

        pub fn eql(_: @This(), a: Type, b: Type) bool {
            return std.meta.eql(a, b);
        }
    };
};

pub const PrimitiveValue = union(enum) {
    type: Type.Id,
    void,
    bool: bool,
    undefined,
    runtime,
    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
    f32: f32,
    f64: f64,
    pointer: Pointer,
    slice: Slice,
    func: Function,
    // TODO: will be a proper pointer later
    string: []const u8,
};

pub const Pointer = extern struct {
    alloc_id: AllocId,
    offset: u32,

    pub const AllocId = enum(u32) {
        _,
    };
};

pub const Slice = extern struct {
    ptr: Pointer,
    len: usize,
};

pub const Function = extern struct {
    entry: Instr.Id,
};

// pub const Value = union {
//     // pub const Id = enum(u32) {
//     //     /// common values
//     //     runtime,
//     //     type_void,
//     //     type_type,
//     //     void,
//     //     bool_true,
//     //     bool_false,
//     //     /// other values
//     //     _,
//     // };

//     type: Type.Id,
//     void: void,
//     bool: bool,
//     int: u64,
//     float: f64,
//     // the data is from the arena allocator and based on the type
//     array: []Value,
//     slice:
//     // compile time pointers are const only (for now)
//     pointer: Value.Id,
//     // func: NodeId,
// };

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
    OutOfMemory,
    OutOfGas,
    Runtime,
};

// strings: std.AutoHashMapUnmanaged(Instr.Index, []const u8) = .{},
reusable_frames: std.SinglyLinkedList = .{},
frames: std.SinglyLinkedList = .{},
types: std.MultiArrayList(Type) = .empty,
type_infos: std.MultiArrayList(TypeInfo) = .empty,
type_map: std.HashMapUnmanaged(Type, Type.Id, Type.Context, 80) = .empty,
arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator),
errors: std.ArrayList(ErrorMsg) = .empty,
error_alloc: std.mem.Allocator = undefined,
ip: Instr.Id = .start,

/// fuel for instruction, null for unlimited
gas: ?usize = null,
/// debug print all frame pop/push events and register set/get events
verbose: bool = false,
mode: EvalMode = .compile,
ir_gen: *const IrGenerator,

pub const builtin_registers: [IrGenerator.Value.builtin_count]Register = [_]Register{
    Register.ty(.type),
    Register.ty(.void),
    Register.ty(.bool),
    Register.ty(.u8),
    Register.ty(.u16),
    Register.ty(.u32),
    Register.ty(.u64),
    Register.ty(._usize),
    Register.ty(.i8),
    Register.ty(.i16),
    Register.ty(.i32),
    Register.ty(.i64),
    Register.ty(._isize),
    Register.ty(.f32),
    Register.ty(.f64),
    Register.ty(._c_int),
    Register.ty(._c_char),
    Register.ty(._c_long),
    Register.ty(._c_longlong),
    Register.ty(._c_short),
    Register.ty(._c_uint),
    Register.ty(._c_ulong),
    Register.ty(._c_ulonglong),
    Register.ty(._c_ushort),
    Register.voidLit(),
    Register.boolLit(false),
    Register.boolLit(true),
    Register.undefinedLit(),
};

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    // self.strings.deinit(alloc);
    while (self.reusable_frames.popFirst()) |first| {
        const frame: *Frame = @fieldParentPtr("node", first);
        frame.registers.deinit(alloc);
        frame.arena.deinit();
        alloc.destroy(frame);
    }
    while (self.frames.popFirst()) |first| {
        const frame: *Frame = @fieldParentPtr("node", first);
        frame.registers.deinit(alloc);
        frame.arena.deinit();
        alloc.destroy(frame);
    }
    self.types.deinit(alloc);
    self.type_infos.deinit(alloc);
    self.type_map.deinit(alloc);
    self.arena.deinit();
    for (self.errors.items) |err| {
        alloc.free(err.message);
    }
    self.errors.deinit(alloc);
}

pub fn printErrors(
    self: *const @This(),
) void {
    const source = self.ir_gen.parser.tokenizer.source;
    for (self.errors.items) |err| {
        const diag = Diagnostic{
            .kind = .err,
            .msg = err,
            .src = .fromSpan(err.span, source),
        };
        std.debug.print("{f}\n", .{diag});
    }
}

pub fn pushError(
    self: *@This(),
    s: Span,
    comptime fmt: []const u8,
    args: anytype,
) Error {
    @branchHint(.cold);
    const message = try std.fmt.allocPrint(self.error_alloc, fmt, args);
    try self.errors.append(self.error_alloc, .{
        .message = message,
        .span = s,
    });
    return Error.Runtime;
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!void {
    self.error_alloc = alloc;

    _ = try self.pushFrame(alloc, .start, .global);

    const instrs = self.ir_gen.instrs.slice();
    const extras = self.ir_gen.extras.items;
    const source = self.ir_gen.parser.tokenizer.source;

    try self.addCommonType(alloc, .type, .type);
    try self.addCommonType(alloc, .void, .void);
    try self.addCommonType(alloc, .bool, .bool);
    try self.addCommonType(alloc, .undefined, .undefined);
    try self.addCommonType(alloc, intType(.unsigned, .@"8"), .u8);
    try self.addCommonType(alloc, intType(.unsigned, .@"16"), .u16);
    try self.addCommonType(alloc, intType(.unsigned, .@"32"), .u32);
    try self.addCommonType(alloc, intType(.unsigned, .@"64"), .u64);
    try self.addCommonType(alloc, intType(.signed, .@"8"), .i8);
    try self.addCommonType(alloc, intType(.signed, .@"16"), .i16);
    try self.addCommonType(alloc, intType(.signed, .@"32"), .i32);
    try self.addCommonType(alloc, intType(.signed, .@"64"), .i64);
    try self.addCommonType(alloc, floatType(.@"32"), .f32);
    try self.addCommonType(alloc, floatType(.@"64"), .f64);
    try self.addCommonType(alloc, .{ .slice = .{
        .mut = false,
        .child = .u8,
    } }, .slice_u8);

    // std.debug.print("VM START\n", .{});
    const gas = if (self.gas) |*_g| _g else null;
    while (true) {
        self.runOnce(
            alloc,
            instrs,
            extras,
            source,
            gas,
        ) catch |err| switch (err) {
            error.OutOfGas => break,
            else => return err,
        };
    }
}

pub fn dump(
    self: *const @This(),
) void {
    std.debug.print("VM DUMP:\n", .{});
    const frame = self.topFrameConst();
    var it = frame.registers.iterator();
    while (it.next()) |reg| {
        std.debug.print("%{} = {any}\n", .{
            @intFromEnum(reg.key_ptr.*),
            reg.value_ptr.*,
        });
    }
}

fn runOnce(
    self: *@This(),
    alloc: std.mem.Allocator,
    instrs: std.MultiArrayList(Instr).Slice,
    extras: []const u32,
    source: []const u8,
    gas: ?*usize,
) Error!void {
    if (gas) |gas_left| {
        if (gas_left.* == 0) return error.OutOfGas;
        gas_left.* -= 1;
    }

    const frame = self.topFrame();
    self.ip = frame.instr;

    const opcode = fetchInstr(instrs, self.ip) orelse {
        @branchHint(.cold);
        return error.OutOfGas;
    };
    frame.instr = @enumFromInt(@intFromEnum(self.ip) + 1);

    if (self.verbose) std.debug.print("exec {f} ({t})\n", .{
        self.ip,
        opcode,
    });
    errdefer if (self.verbose) std.debug.print("error at {f} ({t})\n", .{
        self.ip,
        opcode,
    });
    switch (opcode) {
        .str_lit => |v| try self.set(
            alloc,
            Register.strLit(v.value.read(source)),
        ),
        .int_lit => |v| try self.set(
            alloc,
            Register.intLit(v.value),
        ),
        .float_lit => |v| try self.set(
            alloc,
            Register.floatLit(v.value),
        ),
        .call => |v| {
            const func_reg = try self.get(v.func);
            const proto = self.readType(func_reg.type);
            if (proto != .func) {
                return self.pushError(
                    self.span(self.ip),
                    "'{f}' is not callable",
                    .{func_reg.type.print(self)},
                );
            }

            if (v.argc != proto.func.params.len) {
                return self.pushError(
                    self.span(self.ip),
                    "incorrect argument count: expected {}, found {}",
                    .{ proto.func.params.len, v.argc },
                );
            }

            const args = IrGenerator.Extra.getParams(
                extras,
                v.argv,
                v.argc,
            );
            const node_ids = IrGenerator.Extra.getNodeIds(
                extras,
                v.argv,
                v.argc,
            );

            if (self.mode == .eval) {
                const ip = func_reg.val.func.entry;

                const call_frame = try self.allocFrame(alloc);

                for (args, 0..) |arg, i| {
                    const passed_arg = try self.get(arg.val);
                    if (passed_arg.type != proto.func.params[i]) {
                        return self.pushError(
                            self.ir_gen.parser.node_spans.items[node_ids[i]],
                            "unexpected function argument type: expected '{f}', got: '{f}'",
                            .{
                                proto.func.params[i].print(self),
                                passed_arg.type.print(self),
                            },
                        );
                    }
                    try self.setIndirect(
                        alloc,
                        call_frame,
                        @enumFromInt(@intFromEnum(ip) + i),
                        passed_arg,
                    );
                }

                self.pushAllocatedFrame(ip, call_frame, .local);
                call_frame.symbol = func_reg.name;
                call_frame.return_register = self.ip;
                call_frame.return_type = proto.func.@"return";
                call_frame.function = ip;
                // std.debug.print("{s}\n", .{func_reg.name.read(source)});
            } else {
                for (args, 0..) |arg, i| {
                    const passed_arg = try self.get(arg.val);
                    if (passed_arg.type != proto.func.params[i]) {
                        return self.pushError(
                            self.ir_gen.parser.node_spans.items[node_ids[i]],
                            "unexpected function argument type: expected '{f}', got: '{f}'",
                            .{
                                proto.func.params[i].print(self),
                                passed_arg.type.print(self),
                            },
                        );
                    }
                }
                try self.set(
                    alloc,
                    Register.runtime(proto.func.@"return"),
                );
            }
        },
        .unary_op => |v| {
            const val: Register = try self.get(v.value);
            errdefer {
                std.debug.print("val = {f}; {any}, op = {f}\n", .{
                    val.type.print(self),
                    val.val,
                    v.op,
                });
            }
            const dst: Register = switch (v.op) {
                .slice => Register.ty(try self.resolveType(alloc, .{ .slice = .{
                    .mut = false,
                    .child = try self.regAsType(val),
                } })),
                .slice_mut => Register.ty(try self.resolveType(alloc, .{ .slice = .{
                    .mut = true,
                    .child = try self.regAsType(val),
                } })),
                .pointer => Register.ty(try self.resolveType(alloc, .{ .pointer = .{
                    .mut = false,
                    .child = try self.regAsType(val),
                } })),
                .pointer_mut => Register.ty(try self.resolveType(alloc, .{ .pointer = .{
                    .mut = true,
                    .child = try self.regAsType(val),
                } })),
                .address, .address_mut => {
                    @panic("todo");
                },
                .deref => {
                    @panic("todo");
                },
                inline else => |op| b: {
                    break :b @field(ops, @tagName(op))(val) catch {
                        return self.pushError(
                            self.span(self.ip),
                            "unary op '{f}' not supported for '{f}'",
                            .{ op, val.type.print(self) },
                        );
                    };
                },
            };
            try self.set(alloc, dst);
        },
        .binary_op => |v| {
            const lhs: Register = try self.get(v.lhs);
            const rhs: Register = try self.get(v.rhs);
            errdefer {
                std.debug.print("lhs = {f}; {any}, rhs = {f}; {any}, op = {f}\n", .{
                    lhs.type.print(self),
                    lhs.val,
                    rhs.type.print(self),
                    rhs.val,
                    v.op,
                });
            }
            const dst: Register = switch (v.op) {
                .as => try self.cast(
                    lhs,
                    try self.regAsType(rhs),
                ),
                inline else => |op| b: {
                    break :b @field(ops, @tagName(op))(lhs, rhs) catch {
                        return self.pushError(
                            self.span(self.ip),
                            "binary op '{f}' not supported for '{f}' and '{f}'",
                            .{ op, lhs.type.print(self), rhs.type.print(self) },
                        );
                    };
                },
            };
            try self.set(alloc, dst);
        },
        // .array => {},
        // .alloca => {},
        .write => |v| {
            const target_reg = v.target.asIndex() orelse {
                return self.pushError(
                    self.span(self.ip),
                    "attempt to write to an rvalue",
                    .{},
                );
            };
            const target = try self.getPtr(target_reg);
            target.* = try self.get(v.val);
        },
        // .decl => {},
        .func => |v| {
            const proto = try self.getType(v.proto);
            try self.set(
                alloc,
                Register.func(proto, frame.instr),
            );
            const ip = frame.instr;
            frame.instr = v.body_block_end;

            if (self.mode == .compile) {
                const proto_type = self.readType(proto);
                if (proto_type != .func) {
                    return self.pushError(
                        self.span(self.ip),
                        "type '{f}' is not a function prototype",
                        .{proto.print(self)},
                    );
                }

                const call_frame = try self.pushFrame(alloc, ip, .local);
                call_frame.return_register = null;
                call_frame.return_type = proto_type.func.@"return";
                // std.debug.print("{s}\n", .{func_reg.name.read(source)});

                for (proto_type.func.params, 0..) |param, i| {
                    try self.setIndirect(
                        alloc,
                        call_frame,
                        @enumFromInt(@intFromEnum(ip) + i),
                        Register.runtime(param),
                    );
                }
            }
        },
        .proto => |v| {
            const extra_proto = extras[@intFromEnum(v.extra)..];
            const extra_params = extras[@intFromEnum(v.extra) + 2 ..];
            const proto: IrGenerator.Extra.Proto = @bitCast(extra_proto[0..2].*);
            const params = extra_params[0..proto.param_count];

            const param_types = try alloc.alloc(Type.Id, params.len);
            defer alloc.free(param_types);

            for (params, param_types) |_param, *param_type| {
                const param: IrGenerator.Extra.Param = @bitCast(_param);
                param_type.* = try self.getType(param.val);
            }
            const return_type = try self.getType(proto.return_type);

            const proto_type = try self.resolveType(
                alloc,
                .{ .func = .{
                    .@"extern" = false,
                    .params = param_types,
                    .@"return" = return_type,
                } },
            );
            try self.set(alloc, Register.ty(proto_type));
        },
        .param => {
            // const p = try get(frame, self.ip.asValue());
            // std.debug.print("{any}\n", .{p.val});
        },
        .@"break" => |v| {
            const break_val = try self.get(v.val);

            while (self.topFrame().block != v.block) {
                self.popFrame();
            }
            const break_frame = self.topFrame();
            const return_register = break_frame.return_register;
            const return_type = break_frame.return_type;
            self.popFrame();

            if (return_type) |t| {
                if (break_val.type != t) {
                    return self.pushError(
                        self.span(self.ip),
                        "unexpected type: expected '{f}', found: '{f}'",
                        .{ t.print(self), break_val.type.print(self) },
                    );
                }
            }

            if (return_register) |reg| {
                try self.setIndirect(
                    alloc,
                    self.topFrame(),
                    reg,
                    break_val,
                );
            }
        },
        .@"continue" => |v| {
            while (self.topFrame().block != v.block) {
                self.popFrame();
            }
            if (self.mode == .eval) {
                const new_top_frame = self.topFrame();
                new_top_frame.clear();
            } else {
                self.popFrame();
            }
            // already set by clear: new_top_frame.instr = new_top_frame.block;
        },
        // .dbg_loc => {},
        .dbg_name => |v| {
            var reg = try self.get(v.val);
            reg.name = v.name;
            if (self.mode == .compile) reg.val = .runtime;
            try self.set(alloc, reg);
        },
        .dbg_print => |v| {
            const val = try self.get(v.val);
            if (self.mode == .compile) return;

            switch (val.val) {
                .type => |t| {
                    std.debug.print("{f}\n", .{t.print(self)});
                },
                .void => {
                    std.debug.print("void\n", .{});
                },
                .undefined => {
                    std.debug.print("undefined\n", .{});
                },
                .runtime => {
                    return self.pushError(
                        self.span(self.ip),
                        "cannot print a runtime value at compile time",
                        .{},
                    );
                    // std.debug.print("undefined\n", .{});
                },
                inline .bool, .u8, .u16, .u32, .u64, .i8, .i16, .i32, .i64, .f32, .f64 => |p| {
                    std.debug.print("{}\n", .{p});
                },
                .pointer => |p| {
                    std.debug.print("0x{x}\n", .{@as(u64, @bitCast(p))});
                },
                .slice => {
                    // val.type == .slice_u8;
                    // p.ptr;
                    // std.debug.print(".{}", .{});
                    std.debug.panic("todo\n", .{});
                },
                .func => |p| {
                    std.debug.print("{f}\n", .{p.entry});
                },
                .string => |s| {
                    std.debug.print("{s}\n", .{s});
                },
            }
        },
        .block => |v| {
            const block_frame = try self.pushFrame(
                alloc,
                frame.instr,
                .local_same,
            );
            block_frame.return_register = self.ip;
            frame.instr = v.block_end;
        },
        .conditional => |v| {
            const compile_time_known_branch = try self.getBool(v.boolean);

            if (compile_time_known_branch) |takes_on_true_branch| {
                if (takes_on_true_branch) {
                    const on_true_block_frame = try self.pushFrame(
                        alloc,
                        frame.instr,
                        .local_same,
                    );
                    on_true_block_frame.return_register = self.ip;
                } else {
                    const on_false_block_frame = try self.pushFrame(
                        alloc,
                        v.on_true_block_end,
                        .local_same,
                    );
                    on_false_block_frame.return_register = self.ip;
                }
            } else {
                const on_false_block_frame = try self.pushFrame(
                    alloc,
                    v.on_true_block_end,
                    .local_same,
                );
                on_false_block_frame.return_register = frame.instr;
                const on_true_block_frame = try self.pushFrame(
                    alloc,
                    frame.instr,
                    .local_same,
                );
                on_true_block_frame.return_register = self.ip;
            }

            frame.instr = v.on_false_block_end;
        },
        else => std.debug.panic("TODO: {t}\n", .{opcode}),
    }
}

fn span(
    self: *const @This(),
    instr: Instr.Id,
) Span {
    return self.ir_gen.instr_spans.items[@intFromEnum(instr)];
}

fn set(
    self: *@This(),
    alloc: std.mem.Allocator,
    val: Register,
) Error!void {
    return try self.setIndirect(
        alloc,
        self.topFrame(),
        self.ip,
        val,
    );
}

fn setIndirect(
    self: *@This(),
    alloc: std.mem.Allocator,
    frame: *Frame,
    reg: Instr.Id,
    val: Register,
) Error!void {
    if (self.verbose) std.debug.print("set %{}\n", .{@intFromEnum(self.ip)});
    try frame.registers.putNoClobber(alloc, reg, val);
}

fn get(
    self: *@This(),
    reg: Value,
) Error!Register {
    if (self.verbose) std.debug.print("get {f}\n", .{reg});

    const idx = reg.asIndex() orelse {
        return builtin_registers[@intFromEnum(reg)];
    };

    return (try self.getPtr(idx)).*;
}

fn getPtr(
    self: *@This(),
    reg: Instr.Id,
) Error!*Register {
    if (self.verbose) std.debug.print("getPtr {f}\n", .{reg});

    var frame = self.topFrame();
    const current_function = frame.function;

    while (true) {
        if (frame.registers.getPtr(reg)) |found| {
            if (frame.function != null and frame.function != current_function) {
                return self.pushError(
                    self.span(self.ip),
                    "variable not accessible",
                    .{},
                );
            }
            return found;
        } else {
            @branchHint(.cold);
        }

        // FIXME: handle captures properly
        const next = frame.node.next orelse {
            return self.pushError(
                self.span(self.ip),
                "internal error: register not found",
                .{},
            );
        };
        frame = @fieldParentPtr("node", next);
    }
}

fn getType(
    self: *@This(),
    reg: Value,
) Error!Type.Id {
    const raw = try self.get(reg);
    return try self.regAsType(raw);
}

fn getBool(
    self: *@This(),
    reg: Value,
) Error!?bool {
    const raw = try self.get(reg);
    return try self.regAsBool(raw);
}

fn regAsType(
    self: *@This(),
    reg: Register,
) Error!Type.Id {
    if (reg.type != .type) return {
        return self.pushError(
            self.span(self.ip),
            "unexpected type: expected 'type', found: '{f}'",
            .{reg.type.print(self)},
        );
    };
    return reg.val.type;
}

fn regAsBool(
    self: *@This(),
    reg: Register,
) Error!?bool {
    if (reg.type != .bool) return {
        return self.pushError(
            self.span(self.ip),
            "unexpected type: expected 'bool', found: '{f}'",
            .{reg.type.print(self)},
        );
    };
    if (reg.val == .runtime) return null;
    return reg.val.bool;
}

fn sizeOf(
    self: *const @This(),
    type_id: Type.Id,
) usize {
    const ty = self.typeInfo(type_id);
    return switch (ty) {
        .type => @sizeOf(Type.Id),
        .void => @sizeOf(void),
        .bool => @sizeOf(bool),
        .int => |v| switch (v.bits) {
            .@"8" => @sizeOf(u8),
            .@"16" => @sizeOf(u16),
            .@"32" => @sizeOf(u32),
            .@"64" => @sizeOf(u64),
        },
        .float => |v| switch (v.bits) {
            .@"32" => @sizeOf(f32),
            .@"64" => @sizeOf(f64),
        },
        .array => |v| {
            const elem_size = std.mem.alignForward(usize, self.sizeOf(v.child), self.alignOf(v.child));
            return elem_size * v.len;
        },
        .slice => @sizeOf([]anyopaque),
        .pointer => @sizeOf(*anyopaque),
        .func => @sizeOf(Instr.Id),
    };
}

fn alignOf(
    self: *const @This(),
    type_id: Type.Id,
) usize {
    const ty = self.typeInfo(type_id);
    return switch (ty) {
        .type => @alignOf(Type.Id),
        .void => @alignOf(void),
        .bool => @alignOf(bool),
        .int => |v| switch (v.bits) {
            .@"8" => @alignOf(u8),
            .@"16" => @alignOf(u16),
            .@"32" => @alignOf(u32),
            .@"64" => @alignOf(u64),
        },
        .float => |v| switch (v.bits) {
            .@"32" => @alignOf(f32),
            .@"64" => @alignOf(f64),
        },
        .array => |v| {
            const elem_size = std.mem.alignForward(usize, self.sizeOf(v.child), self.alignOf(v.child));
            return elem_size * v.len;
        },
        .slice => @sizeOf([]anyopaque),
        .pointer => @sizeOf(*anyopaque),
        .func => @sizeOf(Instr.Id),
    };
}

fn PrimitiveOf(
    comptime type_id: Type.Id,
) type {
    return switch (type_id) {
        .u8 => u8,
        .u16 => u16,
        .u32 => u32,
        .u64 => u64,
        .i8 => i8,
        .i16 => i16,
        .i32 => i32,
        .i64 => i64,
        .f32 => f32,
        .f64 => f64,
        .bool => bool,
        .void => void,
        .type => Type.Id,
        else => noreturn,
    };
}

fn primitiveOf(
    comptime type_id: Type.Id,
    val: *anyopaque,
) PrimitiveOf(type_id) {
    return @ptrCast(val);
}

fn topFrame(
    self: *@This(),
) *Frame {
    return frameFromNode(self.frames.first.?);
}

fn topFrameConst(
    self: *const @This(),
) *const Frame {
    return frameFromNode(self.frames.first.?);
}

fn allocFrame(
    self: *@This(),
    alloc: std.mem.Allocator,
) error{OutOfMemory}!*Frame {
    if (self.reusable_frames.popFirst()) |reused| {
        return frameFromNode(reused);
    } else {
        const frame = try alloc.create(Frame);
        frame.* = .{};
        return frame;
    }
}

const FrameKind = enum { global, local, local_same };

fn pushAllocatedFrame(
    self: *@This(),
    ip: Instr.Id,
    frame: *Frame,
    kind: FrameKind,
    // mode: EvalMode,
) void {
    if (self.verbose) std.debug.print("push frame {f}\n", .{ip});

    frame.block = ip;
    frame.instr = ip;
    switch (kind) {
        .local => frame.function = ip,
        .local_same => frame.function = self.topFrame().function,
        else => {},
    }
    // frame.mode = mode;

    self.frames.prepend(&frame.node);
}

fn pushFrame(
    self: *@This(),
    alloc: std.mem.Allocator,
    ip: Instr.Id,
    kind: FrameKind,
    // mode: EvalMode,
) error{OutOfMemory}!*Frame {
    const frame = try self.allocFrame(alloc);
    self.pushAllocatedFrame(ip, frame, kind);
    return frame;
}

fn popFrame(
    self: *@This(),
) void {
    const frame = frameFromNode(self.frames.popFirst().?);
    if (self.verbose) std.debug.print("pop frame {f}\n", .{frame.block});
    frame.clear();
    self.reusable_frames.prepend(&frame.node);
}

fn frameFromNode(node: *std.SinglyLinkedList.Node) *Frame {
    return @fieldParentPtr("node", node);
}

fn fetchInstr(
    instrs: std.MultiArrayList(Instr).Slice,
    instr_ptr: Instr.Id,
) ?Instr {
    const ip: u32 = @intFromEnum(instr_ptr);
    if (ip >= instrs.len) return null;
    return instrs.get(ip);
}

fn addCommonType(
    self: *@This(),
    alloc: std.mem.Allocator,
    ty: Type,
    expected_ty_id: Type.Id,
) Error!void {
    const actual_ty_id = try self.resolveType(alloc, ty);
    std.debug.assert(expected_ty_id == actual_ty_id);
}

fn intType(
    sign: Signedness,
    bits: IntSize,
) Type {
    return .{ .int = .{
        .sign = sign,
        .bits = bits,
    } };
}

fn floatType(
    bits: FloatSize,
) Type {
    return .{ .float = .{
        .bits = bits,
    } };
}

fn intSize(comptime T: type) IntSize {
    return switch (@sizeOf(T)) {
        8 => .@"64",
        4 => .@"32",
        2 => .@"16",
        1 => .@"8",
        else => comptime unreachable,
    };
}

fn resolveType(
    self: *@This(),
    alloc: std.mem.Allocator,
    _ty: Type,
) Error!Type.Id {
    var ty = _ty;

    // std.debug.print("resolve type {}\n", .{ty});
    try self.types.ensureUnusedCapacity(alloc, 1);
    try self.type_infos.ensureUnusedCapacity(alloc, 1);

    const entry = try self.type_map.getOrPut(alloc, ty);
    if (entry.found_existing) return entry.value_ptr.*;

    const idx = self.types.addOneAssumeCapacity();
    std.debug.assert(idx == self.type_infos.addOneAssumeCapacity());
    entry.value_ptr.* = @enumFromInt(idx);

    const type_info: TypeInfo = switch (ty) {
        .type => TypeInfo.of(Type.Id),
        .void => TypeInfo.of(void),
        .bool => TypeInfo.of(bool),
        .undefined => TypeInfo.of(void),
        .int => |v| TypeInfo.ofInt(v.bits),
        .float => |v| TypeInfo.ofFloat(v.bits),
        .array => |v| b: {
            const child = self.readTypeInfo(v.child);
            break :b child.repeat(v.len) orelse {
                return self.pushError(
                    self.span(self.ip),
                    "type too large",
                    .{},
                );
            };
        },
        .slice => TypeInfo.of(Slice),
        .pointer => TypeInfo.of(Pointer),
        .func => b: {
            ty.func.params = try self.arena.allocator().dupe(
                Type.Id,
                ty.func.params,
            );
            break :b TypeInfo.of(Instr.Id);
        },
    };

    self.types.set(idx, ty);
    self.type_infos.set(idx, type_info);
    return @enumFromInt(idx);
}

fn readType(
    self: *const @This(),
    ty: Type.Id,
) Type {
    return self.types.get(@intFromEnum(ty));
}

fn readTypeInfo(
    self: *const @This(),
    ty: Type.Id,
) TypeInfo {
    return self.type_infos.get(@intFromEnum(ty));
}

fn cast(
    self: *@This(),
    _val: Register,
    as: Type.Id,
) Error!Register {
    var val = _val;

    if (val.type == as) {
        return val;
    }

    if (val.type == .undefined) {
        val.type = as;
        return val;
    }

    switch (as) {
        inline .u8, .u16, .u32, .u64, .i8, .i16, .i32, .i64, .f32, .f64 => |into| {
            switch (val.val) {
                inline .u8, .u16, .u32, .u64, .i8, .i16, .i32, .i64, .f32, .f64 => |from| {
                    val.val = @unionInit(
                        PrimitiveValue,
                        @tagName(into),
                        std.math.lossyCast(PrimitiveOf(into), from),
                    );
                    val.type = as;
                    return val;
                },
                else => {},
            }
        },
        else => {},
    }

    return self.pushError(
        self.span(self.ip),
        "cannot cast '{f}' to '{f}'",
        .{ val.type.print(self), as.print(self) },
    );
}

// fn createType(
//     self: *@This(),
//     alloc: std.mem.Allocator,
//     ty: Type,
// ) error{OutOfMemory}!Type.Index {
//     std.debug.print("create type {}\n", .{ty});
//     const type_id: Type.Index = @enumFromInt(try self.types.addOne(alloc));
//     try self.type_map.putNoClobber(alloc, ty, type_id);
//     return type_id;
// }

// fn findType(
//     self: *@This(),
//     ty: Type,
// ) ?Type.Index {
//     return self.type_map.get(ty);
// }

// fn findOrCreateType(
//     self: *@This(),
//     alloc: std.mem.Allocator,
//     ty: Type,
// ) error{OutOfMemory}!Type.Index {
//     const entry = try self.type_map.getOrPut(alloc, ty);
//     // if (entry.found_existing) // copy
//     return entry.value_ptr.*;
// }
