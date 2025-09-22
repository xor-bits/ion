pub const TypeId = u32;

pub const TypeInfo = union(enum) {
    // comptime_int: u128,
    // comptime_float: f64,
    // usize: void,
    // u128: void,
    u64: void,
    u32: void,
    // u16: void,
    u8: void,
    // isize: void,
    // i128: void,
    i64: void,
    i32: void,
    // i16: void,
    i8: void,
    f64: void,
    f32: void,
    bool: void,
    void: void,
    type: void,

    slice: SliceInfo,
    array: ArrayInfo,
    pointer: PointerInfo,
    @"struct": StructInfo,
    @"enum": EnumInfo,
    @"fn": FnInfo,

    pub fn eq(self: @This(), other: @This()) bool {
        // assumes that child types are correctly resolved,
        // so their ids match if they are equal
        return std.meta.eql(self, other);
    }

    pub fn hash(self: @This()) u32 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(@intFromEnum(self));

        switch (self) {
            .comptime_int => |v| hasher.update(std.mem.asBytes(&v)),
            .slice => |v| {
                hasher.update(std.mem.asBytes(&v.elements));
                hasher.update(std.mem.asBytes(&v.mut));
            },
            .array => |v| {
                hasher.update(std.mem.asBytes(&v.elements));
                hasher.update(std.mem.asBytes(&v.len));
            },
            .pointer => |v| {
                hasher.update(std.mem.asBytes(&v.pointee));
                hasher.update(std.mem.asBytes(&v.mut));
            },
            .@"struct" => |v| {
                hasher.update(std.mem.asBytes(&v.fields.start));
                hasher.update(std.mem.asBytes(&v.fields.end));
            },
            .@"enum" => |v| {
                hasher.update(std.mem.asBytes(&v.variants.start));
                hasher.update(std.mem.asBytes(&v.variants.end));
                hasher.update(std.mem.asBytes(&v.tag));
            },
            .@"fn" => |v| {
                for (v.paramns) |param| {
                    hasher.update(std.mem.asBytes(&param.ty));
                }
                hasher.update(std.mem.asBytes(&v.return_ty));
                hasher.update(std.mem.asBytes(&v.@"extern"));
            },
        }

        return @as(u32, @truncate(hasher.final()));
    }
};

pub const SliceInfo = struct {
    elements: TypeId,
    mut: bool,
};

pub const ArrayInfo = struct {
    elements: TypeId,
    len: usize,
};

pub const PointerInfo = struct {
    pointee: TypeId,
    mut: bool,
};

pub const StructInfo = struct {
    fields: []const FieldInfo,
};

pub const EnumInfo = struct {
    variants: []const FieldInfo,
    tag: TypeId,
};

pub const FnInfo = struct {
    paramns: []const FieldInfo,
    return_ty: TypeId,
    @"extern": bool,
};

pub const FieldInfo = struct {
    name: []const u8,
    ty: TypeId,
};

pub const SemanticAnalyzer = struct {
    alloc: std.mem.Allocator,
    types: std.ArrayList(TypeInfo) = .{},
    // types_lookup: std.ArrayHashMap(TypeInfo, TypeId),
    symbols: Symbols = .{},
    ast_types: []TypeId = &.{},

    ast: []const Node,
    source: []const u8,

    pub const Error = error{
        IncompatibleOperands,
        NotAFunction,
        NotAStruct,
        UnknownField,
        UnknownSymbol,
    } || Symbols.Error;

    pub fn init(
        alloc: std.mem.Allocator,
        ast: []const Node,
        source: []const u8,
    ) @This() {
        return .{
            .alloc = alloc,
            .ast = ast,
            .source = source,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.alloc.free(self.ast_types);
        self.symbols.deinit(self.alloc);
        self.types.deinit(self.alloc);
    }

    pub fn run(
        self: *@This(),
    ) Error!void {
        self.ast_types = try self.alloc.alloc(TypeId, self.ast.len);

        try self.symbols.set(self.alloc, "u8", try self.resolveType(.u8));
        try self.symbols.set(self.alloc, "u32", try self.resolveType(.u32));
        try self.symbols.set(self.alloc, "u64", try self.resolveType(.u64));
        try self.symbols.set(self.alloc, "usize", try self.resolveType(.u64));
        try self.symbols.set(self.alloc, "i8", try self.resolveType(.i8));
        try self.symbols.set(self.alloc, "i32", try self.resolveType(.i32));
        try self.symbols.set(self.alloc, "i64", try self.resolveType(.i64));
        try self.symbols.set(self.alloc, "isize", try self.resolveType(.i64));
        try self.symbols.set(self.alloc, "c_int", try self.resolveType(.i32));
        try self.symbols.set(self.alloc, "f32", try self.resolveType(.f32));
        try self.symbols.set(self.alloc, "f64", try self.resolveType(.f64));
        try self.symbols.set(self.alloc, "bool", try self.resolveType(.bool));
        try self.symbols.set(self.alloc, "void", try self.resolveType(.void));
        try self.symbols.set(self.alloc, "type", try self.resolveType(.type));

        try self.convertStructContents(0);
        self.print();
    }

    pub fn print(self: *@This()) void {
        for (self.types.items, 0..) |type_info, type_id| {
            std.debug.print("type_id {}: {}\n", .{ type_id, type_info });
        }

        self.printNode(0, 0);
    }

    fn indent(depth: usize) void {
        for (0..depth) |_| std.debug.print("| ", .{});
    }

    fn printNode(self: *@This(), node: NodeId, depth: usize) void {
        indent(depth);
        // std.debug.print("id={} ", .{node});

        switch (self.ast[node]) {
            .@"struct" => |v| {
                std.debug.print("struct type_id={}:\n", .{self.ast_types[node]});
                self.printNode(v.contents, depth + 1);
            },
            .struct_contents => |v| {
                std.debug.print("struct_contents type_id={}:\n", .{self.ast_types[node]});
                for (v.fields.start..v.fields.end) |i|
                    self.printNode(@truncate(i), depth + 1);
                for (v.decls.start..v.decls.end) |i|
                    self.printNode(@truncate(i), depth + 1);
            },
            .field => |v| {
                std.debug.print("field name='{s}':\n", .{
                    self.readSpan(v.ident),
                });
                self.printNode(v.type, depth + 1);
                if (v.default) |default|
                    self.printNode(default, depth + 1);
            },
            .decl => |v| {
                std.debug.print("decl mut={} name='{s}':\n", .{
                    v.mut,
                    self.readSpan(v.ident),
                });
                self.printNode(v.expr, depth + 1);
            },
            .@"fn" => |v| {
                const type_id = self.ast_types[node];
                const func = self.types.items[type_id].@"fn";
                std.debug.print("fn type_id={} return_type_id={}:\n", .{
                    type_id,
                    func.return_ty,
                });
                indent(depth + 1);
                std.debug.print("proto:\n", .{});
                self.printNode(v.proto, depth + 2);
                indent(depth + 1);
                if (self.ast[v.proto].proto.@"extern")
                    std.debug.print("symexpr:\n", .{})
                else
                    std.debug.print("scope:\n", .{});
                self.printNode(v.scope_or_symexpr, depth + 2);
            },
            .scope => |v| {
                std.debug.print("scope type_id={} autoreturn={}:\n", .{
                    self.ast_types[node],
                    !v.has_trailing_semi,
                });
                for (v.stmts.start..v.stmts.end) |i|
                    self.printNode(@truncate(i), depth + 1);
            },
            .param => |v| {
                std.debug.print("param type_id={} name='{s}':\n", .{
                    self.ast_types[v.type],
                    self.readSpan(v.ident),
                });
                self.printNode(v.type, depth + 1);
            },
            .array => |v| {
                std.debug.print("array type_id={}:\n", .{
                    self.ast_types[node],
                });
                indent(depth + 1);
                std.debug.print("length:\n", .{});
                self.printNode(v.length_expr, depth + 2);
                indent(depth + 1);
                std.debug.print("element:\n", .{});
                self.printNode(v.elements_expr, depth + 2);
            },
            .slice => |v| {
                std.debug.print("slice type_id={} mut={}:\n", .{
                    self.ast_types[node],
                    v.mut,
                });
                self.printNode(v.elements_expr, depth + 1);
            },
            .pointer => |v| {
                std.debug.print("pointer type_id={} mut={}:\n", .{
                    self.ast_types[node],
                    v.mut,
                });
                self.printNode(v.pointee_expr, depth + 1);
            },
            .binary_op => |v| {
                std.debug.print("binary_op type_id={} op={t}:\n", .{
                    self.ast_types[node],
                    v.op,
                });
                self.printNode(v.lhs, depth + 1);
                self.printNode(v.rhs, depth + 1);
            },
            .unary_op => |v| {
                std.debug.print("unary_op type_id={} op={t}:\n", .{
                    self.ast_types[node],
                    v.op,
                });
                self.printNode(v.val, depth + 1);
            },
            .field_acc => |v| {
                std.debug.print("field_acc type_id={} name='{s}':\n", .{
                    self.ast_types[node],
                    self.readSpan(v.ident),
                });
                self.printNode(v.val, depth + 1);
            },
            .index_acc => |v| {
                std.debug.print("index_acc type_id={}:\n", .{
                    self.ast_types[node],
                });
                self.printNode(v.val, depth + 1);
                self.printNode(v.expr, depth + 1);
            },
            .call => |v| {
                std.debug.print("call type_id={}:\n", .{
                    self.ast_types[node],
                });
                self.printNode(v.val, depth + 1);
                indent(depth + 1);
                std.debug.print("args:\n", .{});
                for (v.args.start..v.args.end) |i|
                    self.printNode(@truncate(i), depth + 2);
            },
            .access => |v| {
                std.debug.print("access type_id={}: {s}\n", .{
                    self.ast_types[node],
                    self.readSpan(v.ident),
                });
            },
            .proto => |v| {
                std.debug.print("proto type_id={} extern={}:\n", .{
                    self.ast_types[node],
                    v.@"extern",
                });
                if (v.return_ty_expr) |return_ty_expr| {
                    indent(depth + 1);
                    std.debug.print("return:\n", .{});
                    self.printNode(return_ty_expr, depth + 2);
                }
                indent(depth + 1);
                std.debug.print("params:\n", .{});
                for (v.params.start..v.params.end) |i|
                    self.printNode(@truncate(i), depth + 2);
            },
            .str_lit => |v| {
                std.debug.print("str_lit type_id={}: {s}\n", .{
                    self.ast_types[node],
                    self.readSpan(v.tok),
                });
            },
            .char_lit => |v| {
                std.debug.print("char_lit type_id={} raw={}: {s}\n", .{
                    self.ast_types[node],
                    v.val,
                    self.readSpan(v.tok),
                });
            },
            .int_lit => |v| {
                std.debug.print("int_lit type_id={}: {d}\n", .{
                    self.ast_types[node],
                    v.val,
                });
            },
            .float_lit => |v| {
                std.debug.print("float_lit type_id={}: {d}\n", .{
                    self.ast_types[node],
                    v.val,
                });
            },
        }
    }

    fn builtinTypeString(self: *@This(), ty: union(enum) {
        len: usize,
        slice: void,
        slice_mut: void,
    }) Error!TypeId {
        const char = try self.resolveType(.u8);
        switch (ty) {
            .len => |len| return try self.resolveType(.{ .array = .{
                .elements = char,
                .len = len,
            } }),
            .slice => return try self.resolveType(.{ .slice = .{
                .elements = char,
                .mut = false,
            } }),
            .slice_mut => return try self.resolveType(.{ .slice = .{
                .elements = char,
                .mut = true,
            } }),
        }
    }

    fn resolveType(
        self: *@This(),
        ty: TypeInfo,
    ) Error!TypeId {
        // TODO: hashmap for the real compiler
        for (self.types.items, 0..) |existing, type_id| {
            if (existing.eq(ty)) {
                switch (ty) {
                    .@"fn" => |v| self.alloc.free(v.paramns),
                    else => {},
                }

                return @intCast(type_id);
            }
        }

        // a new type, add it to the list of types
        const type_id: TypeId = @intCast(self.types.items.len);
        try self.types.append(self.alloc, ty);
        return type_id;
    }

    fn readSpan(self: *@This(), span: Span) []const u8 {
        return self.source[span.start..span.end];
    }

    fn convertStructContents(
        self: *@This(),
        id: NodeId,
    ) Error!void {
        self.ast_types[id] = try self.resolveType(.void);
        const contents = self.ast[id].struct_contents;
        for (contents.decls.start..contents.decls.end) |i| {
            try self.convertDecl(@intCast(i));
        }
    }

    fn convertStmt(
        self: *SemanticAnalyzer,
        id: NodeId,
    ) Error!?TypeId {
        switch (self.ast[id]) {
            .decl => {
                try self.convertDecl(id);
                return null;
            },
            else => return try self.convertExpr(id),
        }
    }

    fn convertDecl(
        self: *@This(),
        id: NodeId,
    ) Error!void {
        self.ast_types[id] = try self.resolveType(.void);
        const decl = self.ast[id].decl;
        try self.symbols.set(
            self.alloc,
            self.readSpan(decl.ident),
            try self.convertExpr(decl.expr),
        );
    }

    fn convertExpr(
        self: *@This(),
        id: NodeId,
    ) Error!TypeId {
        switch (self.ast[id]) {
            .array => |v| {},
            .slice => |v| {},
            .pointer => |v| {},
            .binary_op => |v| {
                const type_id_lhs = try self.convertExpr(v.lhs);
                const type_id_rhs = try self.convertExpr(v.rhs);
                if (type_id_lhs != type_id_rhs) return Error.IncompatibleOperands;

                switch (v.op) {
                    .add, .sub, .mul, .div, .rem => {
                        self.ast_types[id] = type_id_lhs;
                        return type_id_lhs;
                    },
                }
            },
            .unary_op => |v| {
                const type_id = try self.convertExpr(v.val);

                switch (v.op) {
                    .not, .neg => {
                        self.ast_types[id] = type_id;
                        return type_id;
                    },
                }
            },
            .call => |v| {
                const type_id = try self.convertExpr(v.val);
                const func = switch (self.types.items[type_id]) {
                    .@"fn" => |s| s,
                    else => return Error.NotAFunction,
                };

                self.ast_types[id] = func.return_ty;
                return func.return_ty;
            },
            .field_acc => |v| {
                const type_id_container = try self.convertExpr(v.val);
                const container = switch (self.types.items[type_id_container]) {
                    .@"struct" => |s| s,
                    else => return Error.NotAStruct,
                };

                for (container.fields) |field| {
                    if (std.mem.eql(u8, field.name, self.readSpan(v.ident))) {
                        const type_id: TypeId = @intCast(field.ty);
                        self.ast_types[id] = type_id;
                        return type_id;
                    }
                }

                return Error.UnknownField;
            },
            .index_acc => |v| {
                _ = v;
                unreachable; // TODO:
            },
            .str_lit => |v| {
                const type_id = try self.builtinTypeString(.{ .len = v.tok.len() - 2 });
                self.ast_types[id] = type_id;
                return type_id;
            },
            .char_lit => {
                const type_id = try self.resolveType(.u8);
                self.ast_types[id] = type_id;
                return type_id;
            },
            .float_lit => |v| {
                const lossy_cast: f32 = @floatCast(v.val);
                const f32_possible = std.math.approxEqAbs(
                    f64,
                    lossy_cast,
                    v.val,
                    std.math.floatEps(f64),
                );

                const type_id = if (f32_possible)
                    try self.resolveType(.f32)
                else
                    try self.resolveType(.f64);
                self.ast_types[id] = type_id;
                return type_id;
            },
            .int_lit => |v| {
                const Int = struct {
                    T: type,
                    info: TypeInfo,
                };
                inline for ([_]Int{
                    .{ .T = i8, .info = .i8 },
                    .{ .T = i64, .info = .i64 },
                    .{ .T = u8, .info = .u8 },
                    .{ .T = u64, .info = .u64 },
                }) |i| {
                    const lossy_cast = std.math.lossyCast(i.T, v.val);
                    if (lossy_cast == v.val) {
                        const type_id = try self.resolveType(i.info);
                        self.ast_types[id] = type_id;
                        return type_id;
                    }
                }
                unreachable;
            },
            .access => |v| {
                const sym = self.readSpan(v.ident);
                const type_id = self.symbols.get(sym) orelse {
                    std.debug.print("unknown symbol: {s}\n", .{sym});
                    return Error.UnknownSymbol;
                };
                self.ast_types[id] = type_id;
                return type_id;
            },
            .scope => return try self.convertScope(id),
            .@"fn" => return try self.convertFn(id),
            else => std.debug.panic("TODO: {}", .{self.ast[id]}),
        }
    }

    fn convertScope(
        self: *SemanticAnalyzer,
        id: NodeId,
    ) Error!TypeId {
        try self.symbols.push(self.alloc);
        defer self.symbols.pop(self.alloc);

        const scope = self.ast[id].scope;
        for (scope.stmts.start..scope.stmts.end) |stmt| {
            _ = try self.convertStmt(@intCast(stmt));
        }

        if (scope.has_trailing_semi or scope.stmts.len() == 0) {
            const type_id = try self.resolveType(.void);
            self.ast_types[id] = type_id;
            return type_id;
        }

        const final_expr = scope.stmts.end - 1;
        if (self.ast[final_expr] == .decl) {
            const type_id = try self.resolveType(.void);
            self.ast_types[id] = type_id;
            return type_id;
        }

        const type_id = self.ast_types[final_expr];
        self.ast_types[id] = type_id;
        return type_id;
    }

    fn convertFn(
        self: *SemanticAnalyzer,
        id: NodeId,
    ) Error!TypeId {
        const f = self.ast[id].@"fn";
        const type_id = try self.convertProto(f.proto);
        self.ast_types[id] = type_id;
        const proto = self.types.items[type_id].@"fn";

        if (proto.@"extern") {
            const type_id_symexpr = try self.convertExpr(f.scope_or_symexpr);
            const type_id_char = switch (self.types.items[type_id_symexpr]) {
                .array => |v| v.elements,
                .slice => |v| v.elements,
                else => return Error.IncompatibleOperands,
            };
            if (self.types.items[type_id_char] != .u8)
                return Error.IncompatibleOperands;

            return type_id;
        }

        try self.symbols.push(self.alloc);
        defer self.symbols.pop(self.alloc);

        for (proto.paramns) |param| {
            try self.symbols.set(
                self.alloc,
                param.name,
                param.ty,
            );
        }

        const type_id_return = try self.convertScope(f.scope_or_symexpr);
        if (type_id_return != proto.return_ty)
            return Error.IncompatibleOperands;

        return type_id;
    }

    fn convertProto(
        self: *@This(),
        id: NodeId,
    ) Error!TypeId {
        const proto = self.ast[id].proto;

        const return_ty = if (proto.return_ty_expr) |return_ty_expr|
            try self.convertExpr(return_ty_expr)
        else
            try self.resolveType(.void);

        for (proto.params.start..proto.params.end) |i| {
            const type_id = try self.convertExpr(self.ast[i].param.type);
            self.ast_types[i] = type_id;
        }

        const params = try self.alloc.alloc(FieldInfo, proto.params.len());
        for (params, proto.params.start..proto.params.end) |*param, i| {
            param.* = .{
                .name = self.readSpan(self.ast[i].param.ident),
                .ty = self.ast_types[i],
            };
        }

        const type_id = try self.resolveType(.{ .@"fn" = .{
            .paramns = params,
            .return_ty = return_ty,
            .@"extern" = proto.@"extern",
        } });
        self.ast_types[id] = type_id;
        return type_id;
    }
};

pub const Symbols = struct {
    scopes: std.ArrayList(std.StringArrayHashMapUnmanaged(TypeId)) = .{},

    pub const Error = error{OutOfMemory};

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
        ty: TypeId,
    ) Error!void {
        try self.lazyinit(alloc);

        const scope = &self.scopes.items[self.scopes.items.len - 1];
        try scope.put(alloc, var_name, ty);
    }

    pub fn get(
        self: *@This(),
        var_name: []const u8,
    ) ?TypeId {
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
