use inkwell::values::BasicValueEnum;

use super::{Compile, Compiler};
use crate::syntax::{BinExpr, BinOp};

//

impl Compile for BinExpr<'_> {
    fn compile<'a>(&mut self, compiler: &mut Compiler<'a>) -> Option<BasicValueEnum<'a>> {
        let lhs = self
            .sides
            .0
            .compile(compiler)
            .expect("LHS returned nothing");
        let rhs = self
            .sides
            .1
            .compile(compiler)
            .expect("RHS returned nothing");

        match (lhs, rhs) {
            (BasicValueEnum::IntValue(lhs), BasicValueEnum::IntValue(rhs)) => {
                let b = &compiler.builder;
                let n = "int bin expr";
                Some(
                    match self.op {
                        BinOp::Add => b.build_int_add(lhs, rhs, n),
                        BinOp::Sub => b.build_int_sub(lhs, rhs, n),
                        BinOp::Mul => b.build_int_mul(lhs, rhs, n),
                        BinOp::Div => b.build_int_signed_div(lhs, rhs, n),
                    }
                    .into(),
                )
            }
            (BasicValueEnum::FloatValue(lhs), BasicValueEnum::FloatValue(rhs)) => {
                let b = &compiler.builder;
                let n = "float bin expr";
                Some(
                    match self.op {
                        BinOp::Add => b.build_float_add(lhs, rhs, n),
                        BinOp::Sub => b.build_float_sub(lhs, rhs, n),
                        BinOp::Mul => b.build_float_mul(lhs, rhs, n),
                        BinOp::Div => b.build_float_div(lhs, rhs, n),
                    }
                    .into(),
                )
            }
            (lhs, rhs) => {
                panic!("cannot run op {:?} with {lhs} and {rhs}", self.op)
            }
        }
    }
}