const std = @import("std");

const Tokenizer = @import("Tokenizer.zig");
const Span = Tokenizer.Span;
const Parser = @import("Parser.zig");
const Node = Parser.Node;
const NodeId = Parser.NodeId;
const NameHint = @import("main.zig").NameHint;

pub const Variable = struct {
    ident: Span,
    version: u32,
    depth: u8,
};

pub const BuiltinType = enum {
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

pub const Error = error{
    OutOfMemory,
    WriteFailed,
    VariableNotFound,
};

variable_version_map: std.MultiArrayList(Variable) = .{},
depth: u8 = 0,
scope_id: u32 = 0,

parser: *Parser,
destin_file: std.fs.File,

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    self.variable_version_map.deinit(alloc);
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

fn pushScope(
    self: *@This(),
) void {
    self.depth += 1;
}

fn popScope(
    self: *@This(),
) void {
    // capacity is not changed so the pointer should stay stable
    const depths = self.variable_version_map.items(.depth);
    var n = depths.len;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        std.debug.assert(depths.ptr == self.variable_version_map.items(.depth).ptr);
        if (depths[i] != self.depth) continue;

        self.variable_version_map.swapRemove(i);
        n -= 1;
    }
    self.depth -= 1;
}

fn findVariable(
    self: *@This(),
    name: Span,
) Error!Variable {
    const name_str = name.read(self.source());

    const slice = self.variable_version_map.slice();
    const idents = slice.items(.ident);
    const depths = slice.items(.depth);

    var found = false;
    var idx: usize = 0;
    var max_depth: u8 = 0;

    std.debug.print("find {s}\n", .{
        name_str,
    });

    if (null != std.meta.stringToEnum(BuiltinType, name_str)) return .{
        .ident = name,
        .depth = 0,
        .version = 0,
    };

    for (0..slice.len) |i| {
        if (!std.mem.eql(u8, idents[i].read(self.source()), name_str)) continue;
        const depth = depths[i];
        if (depth < max_depth) continue;

        found = true;
        max_depth = depth;
        idx = i;
    }

    if (!found) return Error.VariableNotFound;

    return slice.get(idx);
}

fn createVariable(
    self: *@This(),
    alloc: std.mem.Allocator,
    name: Span,
) Error!Variable {
    const new: Variable = if (self.findVariable(name)) |previous| .{
        .ident = name,
        .depth = self.depth,
        .version = previous.version + 1,
    } else |_| .{
        .ident = name,
        .depth = self.depth,
        .version = 0,
    };

    std.debug.print("create {s}_{}\n", .{
        new.ident.read(self.source()),
        new.version,
    });

    try self.variable_version_map.append(alloc, new);
    return new;
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) Error!void {
    var write_buffer: [0x8000]u8 = undefined;
    var source_writer = self.destin_file.writer(&write_buffer);
    const writer = &source_writer.interface;

    for (std.enums.values(BuiltinType)) |builtin| {
        try writer.print("const {0t}_0 = {0t};\n", .{
            builtin,
        });
    }
    try writer.print("pub const main = main_0;\n", .{});

    _ = try self.convertStructContents(
        alloc,
        writer,
        &.{ .part = "<root>" },
        0,
    );

    try writer.flush();
}

pub fn convertStructContents(
    self: *@This(),
    alloc: std.mem.Allocator,
    writer: *std.io.Writer,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const struct_contents = self.nodes()[node_id].struct_contents;

    for (struct_contents.decls.start..struct_contents.decls.end) |i| {
        try self.convertDecl(
            alloc,
            writer,
            name_hint,
            @intCast(i),
            .global,
        );
        try writer.print(";\n", .{});
    }
}

pub fn convertDecl(
    self: *@This(),
    alloc: std.mem.Allocator,
    writer: *std.io.Writer,
    name_hint: *const NameHint,
    node_id: NodeId,
    mode: enum { local, global },
) Error!void {
    const decl = self.nodes()[node_id].decl;
    const name = decl.ident.read(self.source());

    const variable = try self.createVariable(alloc, decl.ident);

    if (decl.mut) {
        try writer.print("{s}var ", .{
            if (mode == .global) "pub " else "",
        });
    } else {
        try writer.print("{s}const ", .{
            if (mode == .global) "pub " else "",
        });
    }

    try writer.print("{s}_{}", .{
        name,
        variable.version,
    });

    if (decl.type_hint) |i| {
        try writer.print(": ", .{});
        try self.convertExpr(
            alloc,
            writer,
            &name_hint.push(name),
            i,
        );
    }

    try writer.print(" = ", .{});

    try self.convertExpr(
        alloc,
        writer,
        &name_hint.push(name),
        decl.expr,
    );

    if (mode == .local)
        try writer.print(";\n_ = .{{{s}_{}}}", .{
            name,
            variable.version,
        });
}

pub fn convertExpr(
    self: *@This(),
    alloc: std.mem.Allocator,
    writer: *std.io.Writer,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    try writer.print("(", .{});

    switch (self.nodes()[node_id]) {
        .array => |v| {
            try writer.print("[", .{});
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                v.length_expr,
            );
            try writer.print("]", .{});
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                v.elements_expr,
            );
        },
        .slice => |v| {
            if (v.mut) {
                try writer.print("[]", .{});
            } else {
                try writer.print("[]const ", .{});
            }
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                v.elements_expr,
            );
        },
        .pointer => |v| {
            if (v.mut) {
                try writer.print("*", .{});
            } else {
                try writer.print("*const ", .{});
            }
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                v.pointee_expr,
            );
        },
        .binary_op => |v| {
            if (v.op == .as) {
                try writer.print("@ptrCast(", .{});
                try self.convertExpr(
                    alloc,
                    writer,
                    name_hint,
                    v.lhs,
                );
                try writer.print(")", .{});
                // TODO: rhs
            } else {
                try self.convertExpr(
                    alloc,
                    writer,
                    name_hint,
                    v.lhs,
                );
                try writer.print("{s}", .{switch (v.op) {
                    .add => "+",
                    .sub => "-",
                    .mul => "*",
                    .div => "/",
                    .rem => "%",
                    .as => unreachable,
                }});
                try self.convertExpr(
                    alloc,
                    writer,
                    name_hint,
                    v.rhs,
                );
            }
        },
        .unary_op => |v| {
            try writer.print("{s}", .{switch (v.op) {
                .neg => "~",
                .not => "!",
            }});
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                v.val,
            );
        },
        .call => |v| {
            try writer.print("(", .{});
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                v.val,
            );
            try writer.print(")", .{});

            try writer.print("(", .{});
            for (v.args.start..v.args.end) |i| {
                try self.convertExpr(
                    alloc,
                    writer,
                    name_hint,
                    @intCast(i),
                );
                try writer.print(",", .{});
            }
            try writer.print(")", .{});
        },
        .field_acc => |v| {
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                v.val,
            );
            try writer.print(".{s}", .{
                v.ident.read(self.source()),
            });
        },
        .index_acc => |v| {
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                v.val,
            );
            try writer.print("[", .{});
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                v.expr,
            );
            try writer.print("]", .{});
        },
        .str_lit => |v| {
            try writer.print("{s}", .{
                v.tok.read(self.source()),
            });
        },
        .char_lit => |v| {
            try writer.print("{}", .{
                v.val,
            });
        },
        .float_lit => |v| {
            try writer.print("{}", .{
                v.val,
            });
        },
        .int_lit => |v| {
            try writer.print("{}", .{
                v.val,
            });
        },
        .access => |v| {
            const variable = try self.findVariable(v.ident);
            try writer.print("{s}_{}", .{
                variable.ident.read(self.source()),
                variable.version,
            });
        },
        .scope => try self.convertScope(
            alloc,
            writer,
            &name_hint.push("<scope>"),
            node_id,
        ),
        .@"fn" => try self.convertFn(
            alloc,
            writer,
            name_hint,
            node_id,
        ),
        .proto => try self.convertProto(
            alloc,
            writer,
            name_hint,
            node_id,
        ),
        else => std.debug.panic("TODO: {}", .{self.nodes()[node_id]}),
    }

    try writer.print(")", .{});
}

pub fn convertScope(
    self: *@This(),
    alloc: std.mem.Allocator,
    writer: *std.io.Writer,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const scope = self.nodes()[node_id].scope;

    const scope_id = self.scope_id;
    try writer.print("_{}: {{", .{
        scope_id,
    });
    self.scope_id += 1;

    if (scope.stmts.splitLast()) |_stmts| {
        const stmts = _stmts.@"0";
        const last_stmt = _stmts.@"1";

        for (stmts.start..stmts.end) |i| {
            try self.convertStmt(
                alloc,
                writer,
                name_hint,
                @intCast(i),
            );
        }

        if (!scope.has_trailing_semi)
            try writer.print("break :_{} ", .{
                scope_id,
            });
        try self.convertStmt(
            alloc,
            writer,
            name_hint,
            last_stmt,
        );
        if (scope.has_trailing_semi)
            try writer.print("break :_{} {{}};", .{
                scope_id,
            });
    } else {
        try writer.print("break :_{} {{}};", .{
            scope_id,
        });
    }

    try writer.print("}}", .{});
}

pub fn convertStmt(
    self: *@This(),
    alloc: std.mem.Allocator,
    writer: *std.io.Writer,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    switch (self.nodes()[node_id]) {
        .decl => {
            try self.convertDecl(
                alloc,
                writer,
                name_hint,
                node_id,
                .local,
            );
        },
        else => {
            try self.convertExpr(
                alloc,
                writer,
                name_hint,
                node_id,
            );
        },
    }
    try writer.print(";\n", .{});
}

pub fn convertFn(
    self: *@This(),
    alloc: std.mem.Allocator,
    writer: *std.io.Writer,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const func = self.nodes()[node_id].@"fn";
    const proto = self.nodes()[func.proto].proto;

    const tmp_name = if (proto.@"extern") b: {
        // TODO: evaluate the symbol name, Zig doesn't support this afaik
        const sym_name_str = self.nodes()[func.scope_or_symexpr].str_lit.tok.read(self.source());
        const sym_name = sym_name_str[1 .. sym_name_str.len - 1];

        break :b sym_name;
    } else "f";

    try writer.print("(struct {{ {s}fn {s}(", .{
        if (proto.@"extern") "extern " else "",
        tmp_name,
    });

    self.pushScope();
    defer self.popScope();

    for (proto.params.start..proto.params.end) |i| {
        const param = self.nodes()[i].param;

        const param_variable = try self.createVariable(
            alloc,
            param.ident,
        );

        try writer.print("{s}_{}: ", .{
            param.ident.read(self.source()),
            param_variable.version,
        });
        try self.convertExpr(
            alloc,
            writer,
            name_hint,
            param.type,
        );
        try writer.print(",", .{});
    }
    if (proto.is_va_args) {
        try writer.print("...", .{});
    }

    try writer.print(") ", .{});

    if (proto.return_ty_expr) |i| {
        try self.convertExpr(
            alloc,
            writer,
            name_hint,
            i,
        );
    } else {
        try writer.print("void", .{});
    }

    if (!proto.@"extern") {
        try writer.print(" {{\n", .{});

        try writer.print("const @\"result\" =\n", .{});
        try self.convertScope(
            alloc,
            writer,
            name_hint,
            func.scope_or_symexpr,
        );
        try writer.print(";\nreturn @\"result\";\n", .{});

        try writer.print("}} }}).{s}", .{
            tmp_name,
        });
    } else {
        try writer.print(";\n}}).{s}", .{
            tmp_name,
        });
    }
}

const s = (struct {
    pub extern fn printf(s: *const u8, ...) c_int;
}).printf;

pub fn convertProto(
    self: *@This(),
    alloc: std.mem.Allocator,
    writer: *std.io.Writer,
    name_hint: *const NameHint,
    node_id: NodeId,
) Error!void {
    const proto = self.nodes()[node_id].proto;

    self.pushScope();
    defer self.popScope();

    for (proto.params.start..proto.params.end) |i| {
        const param = self.nodes()[i].param;

        const param_variable = try self.createVariable(
            alloc,
            param.ident,
        );

        try writer.print("{s}_{}: ", .{
            param.ident.read(self.source()),
            param_variable.version,
        });
        try self.convertExpr(
            alloc,
            writer,
            name_hint,
            param.type,
        );
    }

    try writer.print(") ", .{});

    if (proto.return_ty_expr) |i| {
        try self.convertExpr(
            alloc,
            writer,
            name_hint,
            i,
        );
    } else {
        try writer.print("void", .{});
    }
}
