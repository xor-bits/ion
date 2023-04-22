use serde::Serialize;

use super::{Ident, ToStatic};
use crate::ty::IonType;

//

/// Data accessor
#[derive(Debug, Clone, ToStatic, Serialize)]
pub struct Path<'i> {
    pub ty: IonType,
    pub parts: Vec<Ident<'i>>,
}

impl<'i> Path<'i> {
    pub fn new(parts: Vec<Ident<'i>>) -> Self {
        Self {
            ty: IonType::Unknown,
            parts,
        }
    }
}
