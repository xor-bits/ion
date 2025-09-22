// const llvm = @cImport({
//     @cInclude("llvm-c/Core.h");
//     @cInclude("llvm-c/Analysis.h");
// });

// _ = llvm.LLVMCreateBuilder();
// const mod = llvm.LLVMModuleCreateWithName("ion");
// const main_fn_ty = llvm.LLVMFunctionType(llvm.LLVMInt32Type(), null, 0, 0);
// _ = llvm.LLVMAddFunction(mod, "main", main_fn_ty);
// var err: [*c]u8 = null;
// _ = llvm.LLVMVerifyModule(mod, llvm.LLVMAbortProcessAction, &err);
// llvm.LLVMDumpModule(mod);

pub const Codegen = struct {
    alloc: std.mem.Allocator,
    globals: std.ArrayList(u8) = .{},

    ast: []const Node,
    source: []const u8,

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
        self.globals.deinit(self.alloc);
    }

    pub fn run(self: *@This()) !void {
        _ = self; // autofix
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
};
