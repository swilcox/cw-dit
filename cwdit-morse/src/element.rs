//! Morse-element and gap types.
//!
//! A key-down interval classifies as an [`Element`] (dit or dah). A key-up
//! interval classifies as a [`Gap`] separating elements, characters, or words.

/// A key-down Morse element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    /// Short mark ("dit"), nominal duration = 1 unit.
    Dit,
    /// Long mark ("dah"), nominal duration = 3 units.
    Dah,
}

impl Element {
    /// ASCII glyph: `'.'` for [`Element::Dit`], `'-'` for [`Element::Dah`].
    #[must_use]
    pub const fn glyph(self) -> char {
        match self {
            Self::Dit => '.',
            Self::Dah => '-',
        }
    }
}

/// A key-up interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gap {
    /// Gap between elements inside a character (nominal 1 unit).
    IntraChar,
    /// Gap between characters (nominal 3 units).
    Char,
    /// Gap between words (nominal 7 units).
    Word,
}

/// One event produced by the decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Decoded {
    /// A character was decoded from an accumulated pattern.
    Char(char),
    /// The accumulated pattern did not match any known character.
    Unknown,
    /// A word boundary was detected.
    WordBreak,
}
