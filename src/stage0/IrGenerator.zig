const std = @import("std");
const Parser = @import("Parser.zig");
const Tokenizer = @import("Tokenizer.zig");
const Span = Tokenizer.Span;

pub const InstrId = struct { u32 };
pub const RegId = struct { u32 };

pub const Instr = union(enum) {
    /// create and initialize a variable
    let: struct {
        target: RegId,
        ident: Span,
        mut: bool,
    },
    /// reads the address of a field `a.b`
    get_field: struct {
        target: RegId,
        container: RegId,
        field_idx: u32,
    },
    /// always followed by another `.call_arg` or `.call`
    call_arg: struct {
        value: RegId,
    },
    /// calls a function `a()` with arguments given by earlier `.call_arg`s
    call: struct {
        func: RegId,
    },
};

pub const Function = struct {
    
};

instr: std.MultiArrayList(Instr) = .{},
functions: std.MultiArrayList(Function) = .{},
operands: std.ArrayList(IrOperands) = .{},
global_scope: InstrId = 0,
parser: Parser,

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

pub fn deinit(
    self: *@This(),
    alloc: std.mem.Allocator,
) void {
    self.operands.deinit(self.alloc);
    self.instr.deinit(self.alloc);
}

pub fn run(self: *@This()) !void {
    // measure the avg ir instruction count
    try self.instr.ensureTotalCapacity(self.alloc, 16);
    try self.operands.ensureTotalCapacity(self.alloc, 16);

    self.convertStructContents(self.ast[0].struct_contents);
}

pub fn convertStructContents(
    self: *@This(),
    struct_contents: @FieldType(Node, "struct_contents"),
) void {
    struct_contents.decls;
    struct_contents.fields;
    _ = self; // autofix
    _ = struct_contents; // autofix

}
