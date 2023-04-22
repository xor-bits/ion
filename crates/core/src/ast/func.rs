use std::borrow::Cow;

use serde::Serialize;

use super::{Block, Expr, Ident, ToStatic};
use crate::ty::IonType;

//

/// Callable function
///
/// All Ion functions are like 'lamdas' or 'closures'.
/// They might or might not be just structs with generic methods.
///
/// This allows this statically typed code:
/// ```ignore
/// let x = fn(p) { return p + p; };
///
/// print(x(2));   // prints 4
/// print(x("2")); // prints 22
/// ```
///
/// This basically compiles down to something like this:
/// ```
/// let x = <nameless fn struct #xyz>;
///
/// print(<nameless fn struct #xyz (int)>::call(2));
/// print(<nameless fn struct #xyz (str)>::call("2"));
/// ```
///
/// TODO: extern functions (bodyless)
#[derive(Debug, Clone, ToStatic, Serialize)]
pub struct Fn<'i> {
    pub ty: IonType,
    pub name: Ident<'i>,
    /// the first param hides all captured variables OR it is a void
    /// first param is the struct object, params after it are these:
    pub params: Vec<(IonType, Ident<'i>)>,
    pub block: Block<'i>,
}

impl<'i> Fn<'i> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(mut self, name: impl Into<Ident<'i>>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_random_name(mut self) -> Self {
        // TODO: more context
        let id: usize = rand::random();
        self.name = Cow::Owned(format!("nameless function {id}"));
        self
    }

    pub fn with_params(mut self, params: Vec<Ident<'i>>) -> Self {
        self.params = params
            .into_iter()
            .map(|ident| (IonType::Unknown, ident))
            .collect();
        self
    }

    pub fn with_block(mut self, block: Block<'i>) -> Self {
        self.block = block;
        self
    }
}

impl Default for Fn<'_> {
    fn default() -> Self {
        Self {
            ty: IonType::Unknown,
            name: Ident::Borrowed("incomplete function"),
            params: Default::default(),
            block: Default::default(),
        }
    }
}

#[derive(Debug, Clone, ToStatic, Serialize)]
pub struct FnCall<'i> {
    pub ty: IonType,
    pub fn_ty: IonType,
    pub name: Ident<'i>,
    pub args: Vec<Expr<'i>>,
}

impl<'i> FnCall<'i> {
    pub fn new(name: Ident<'i>, args: Vec<Expr<'i>>) -> Self {
        Self {
            ty: <_>::default(),
            fn_ty: <_>::default(),
            name,
            args,
        }
    }
}

#[derive(Debug, Clone, ToStatic, Serialize)]
pub struct Return<'i> {
    pub value: Expr<'i>,
}

#[derive(Debug, Clone, ToStatic, Default, Serialize)]
#[to_static(result = "Self")]
pub struct ReturnVoid;
