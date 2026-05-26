const std = @import("std");

const Register = @import("../VirtualMachine.zig").Register;
const PrimitiveValue = @import("../VirtualMachine.zig").PrimitiveValue;
const Type = @import("../VirtualMachine.zig").Type;
const Error = @import("../VirtualMachine.zig").Error;

pub fn add(lhs: Register, rhs: Register) Error!Register {
    return try mathOp(addIntFunc, addFloatFunc, lhs, rhs);
}

fn addIntFunc(l: anytype, r: @TypeOf(l)) @TypeOf(l) {
    return l +% r;
}

fn addFloatFunc(l: anytype, r: @TypeOf(l)) @TypeOf(l) {
    return l + r;
}

pub fn sub(lhs: Register, rhs: Register) Error!Register {
    return try mathOp(subIntFunc, subFloatFunc, lhs, rhs);
}

fn subIntFunc(l: anytype, r: @TypeOf(l)) @TypeOf(l) {
    return l -% r;
}

fn subFloatFunc(l: anytype, r: @TypeOf(l)) @TypeOf(l) {
    return l - r;
}

pub fn mul(lhs: Register, rhs: Register) Error!Register {
    return try mathOp(mulIntFunc, mulFloatFunc, lhs, rhs);
}

fn mulIntFunc(l: anytype, r: @TypeOf(l)) @TypeOf(l) {
    return l * r;
}

fn mulFloatFunc(l: anytype, r: @TypeOf(l)) @TypeOf(l) {
    return l * r;
}

pub fn div(lhs: Register, rhs: Register) Error!Register {
    return try mathOp(divFunc, divFunc, lhs, rhs);
}

fn divFunc(l: anytype, r: @TypeOf(l)) @TypeOf(l) {
    return @divTrunc(l, r);
}

pub fn mod(lhs: Register, rhs: Register) Error!Register {
    return try mathOp(modFunc, modFunc, lhs, rhs);
}

fn modFunc(l: anytype, r: @TypeOf(l)) @TypeOf(l) {
    return @mod(l, r);
}

pub fn as(lhs: Register, rhs: Register) Error!Register {
    _ = lhs;
    _ = rhs;
    std.debug.panic("TODO: as op", .{});
}

pub fn eq(lhs: Register, rhs: Register) Error!Register {
    return try boolOp(eqFunc, lhs, rhs);
}

fn eqFunc(l: anytype, r: @TypeOf(l)) bool {
    return l == r;
}

pub fn neq(lhs: Register, rhs: Register) Error!Register {
    return try boolOp(neqFunc, lhs, rhs);
}

fn neqFunc(l: anytype, r: @TypeOf(l)) bool {
    return l != r;
}

pub fn lt(lhs: Register, rhs: Register) Error!Register {
    return try boolOp(ltFunc, lhs, rhs);
}

fn ltFunc(l: anytype, r: @TypeOf(l)) bool {
    return l < r;
}

pub fn le(lhs: Register, rhs: Register) Error!Register {
    return try boolOp(leFunc, lhs, rhs);
}

fn leFunc(l: anytype, r: @TypeOf(l)) bool {
    return l <= r;
}

pub fn gt(lhs: Register, rhs: Register) Error!Register {
    return try boolOp(gtFunc, lhs, rhs);
}

fn gtFunc(l: anytype, r: @TypeOf(l)) bool {
    return l > r;
}

pub fn ge(lhs: Register, rhs: Register) Error!Register {
    return try boolOp(geFunc, lhs, rhs);
}

fn geFunc(l: anytype, r: @TypeOf(l)) bool {
    return l >= r;
}

pub fn field(lhs: Register, rhs: Register) Error!Register {
    _ = lhs;
    _ = rhs;
    std.debug.panic("TODO: field op", .{});
}

pub fn index(lhs: Register, rhs: Register) Error!Register {
    _ = lhs;
    _ = rhs;
    std.debug.panic("TODO: index op", .{});
}

pub fn range(lhs: Register, rhs: Register) Error!Register {
    _ = lhs;
    _ = rhs;
    std.debug.panic("TODO: range op", .{});
}

pub fn neg(val: Register) Error!Register {
    return try unaryOp(negIntFunc, negFloatFunc, val);
}

fn negIntFunc(val: anytype) @TypeOf(val) {
    return -%val;
}

fn negFloatFunc(val: anytype) @TypeOf(val) {
    return -val;
}

pub fn not(val: Register) Error!Register {
    if (val.val != .bool) return Error.OperationUnsupportedForType;
    return .{ .type = .bool, .val = .{
        .bool = !val.val.bool,
    } };
}

// neg,
// not,
// slice,
// slice_mut,
// pointer,
// pointer_mut,
// address,
// address_mut,
// deref,

pub fn slice(val: Register) Error!Register {
    if (val.val != .bool) return Error.OperationUnsupportedForType;
    return .{ .type = .bool, .val = .{
        .bool = !val.val.bool,
    } };
}

fn mathOp(
    comptime funcInt: anytype,
    comptime funcFloat: anytype,
    lhs: Register,
    rhs: Register,
) Error!Register {
    if (lhs.type != rhs.type) return Error.TypeMismatch;
    std.debug.assert(std.meta.activeTag(lhs.val) == rhs.val);
    return switch (std.meta.activeTag(lhs.val)) {
        inline .u8, .u16, .u32, .u64, .i8, .i16, .i32, .i64 => |v| {
            const t = @tagName(v);
            return .{ .type = lhs.type, .val = @unionInit(
                PrimitiveValue,
                t,
                funcInt(
                    @field(lhs.val, t),
                    @field(rhs.val, t),
                ),
            ) };
        },
        inline .f32, .f64 => |v| {
            const t = @tagName(v);
            return .{ .type = lhs.type, .val = @unionInit(
                PrimitiveValue,
                t,
                funcFloat(
                    @field(lhs.val, t),
                    @field(rhs.val, t),
                ),
            ) };
        },
        else => Error.OperationUnsupportedForType,
    };
}

fn boolOp(
    comptime func: anytype,
    lhs: Register,
    rhs: Register,
) Error!Register {
    if (lhs.type != rhs.type) return Error.TypeMismatch;
    std.debug.assert(std.meta.activeTag(lhs.val) == rhs.val);
    return switch (std.meta.activeTag(lhs.val)) {
        inline .u8, .u16, .u32, .u64, .i8, .i16, .i32, .i64, .f32, .f64 => |v| {
            const t = @tagName(v);
            return .{ .type = .bool, .val = .{ .bool = func(
                @field(lhs.val, t),
                @field(rhs.val, t),
            ) } };
        },
        else => Error.OperationUnsupportedForType,
    };
}

fn unaryOp(
    comptime funcInt: anytype,
    comptime funcFloat: anytype,
    val: Register,
) Error!Register {
    return switch (std.meta.activeTag(val.val)) {
        inline .u8, .u16, .u32, .u64, .i8, .i16, .i32, .i64 => |v| {
            const t = @tagName(v);
            return .{ .type = val.type, .val = @unionInit(
                PrimitiveValue,
                t,
                funcInt(@field(val.val, t)),
            ) };
        },
        inline .f32, .f64 => |v| {
            const t = @tagName(v);
            return .{ .type = val.type, .val = @unionInit(
                PrimitiveValue,
                t,
                funcFloat(@field(val.val, t)),
            ) };
        },
        else => Error.OperationUnsupportedForType,
    };
}
