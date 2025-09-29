const std = @import("std");

const Parser = @import("Parser.zig");
const Node = Parser.Node;
const NodeId = Parser.NodeId;

const llvm = @cImport({
    @cInclude("llvm-c/Core.h");
    @cInclude("llvm-c/Analysis.h");
});

// _ = llvm.LLVMCreateBuilder();
// const mod = llvm.LLVMModuleCreateWithName("ion");
// const main_fn_ty = llvm.LLVMFunctionType(llvm.LLVMInt32Type(), null, 0, 0);
// _ = llvm.LLVMAddFunction(mod, "main", main_fn_ty);
// var err: [*c]u8 = null;
// _ = llvm.LLVMVerifyModule(mod, llvm.LLVMAbortProcessAction, &err);
// llvm.LLVMDumpModule(mod);

pub const Value = union(enum) {
    runtime: llvm.LLVMValueRef,
    // comptime_
    // undef: void,
};

globals: std.ArrayList(u8) = .{},
symbols: Symbols = .{},

ast: []const Node,
source: []const u8,

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    self.globals.deinit(alloc);
}

pub fn run(
    self: *@This(),
    alloc: std.mem.Allocator,
) !void {
    _ = try self.convertStructContents(alloc, self.ast[0].struct_contents);
}

pub fn convertStructContents(
    self: *@This(),
    alloc: std.mem.Allocator,
    ast: @FieldType(Node, "struct_contents"),
) !Value {
    for (ast.decls.start..ast.decls.end) |i| {
        const decl = self.ast[i].decl;
        const var_name = decl.ident.read(self.source);
        const value = try self.convertDecl(alloc, decl);

        try self.symbols.set(alloc, var_name, value);
    }

    // for (ast.decls.start..ast.decls.end) |i| {
    //     self.symbols.set(alloc, var_name, val: Value);
    // }

    // ast.decls;
    // ast.fields;

    unreachable;
}

pub fn convertDecl(
    self: *@This(),
    alloc: std.mem.Allocator,
    ast: @FieldType(Node, "decl"),
) !Value {
    _ = alloc;
    switch (self.ast[ast.expr]) {
        else => |s| std.debug.panic("unhandled: {}", .{s}),
    }
}

pub fn convertFn(
    self: *@This(),
    alloc: std.mem.Allocator,
    ast: @FieldType(Node, "fn"),
) !Value {
    const proto = try self.convertProto(alloc, self.ast[ast.proto].proto);
    _ = proto;
}

pub fn convertProto(
    self: *@This(),
    alloc: std.mem.Allocator,
    ast: @FieldType(Node, "proto"),
) !Value {
    ast.@"extern";
    ast.is_va_args;
    ast.params;
    ast.return_ty_expr;

    ast.params;

    for (ast.params.start..ast.params.end) |i| {
        self.ast[i].param;
    }

    self.ast[ast.return_ty_expr];
}

pub fn dump(
    self: *@This(),
    ir_writer: *std.io.Writer,
) !void {
    _ = self;
    _ = try ir_writer.write(
        \\declare i32 @puts(i8*) nounwind
        \\@.hello = private unnamed_addr constant [14 x i8] c"Hello, world!\00"
        \\define i32 @main(i32 %argc, i8** %argv) nounwind {
        \\    %1 = getelementptr [14 x i8], [14 x i8]* @.hello, i32 0, i32 0
        \\    call i32 @puts(i8* %1)
        \\    ret i32 0
        \\}
        \\
    );
}
