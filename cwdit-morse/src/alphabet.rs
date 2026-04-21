//! International Morse alphabet — `char` ⇄ pattern conversions.
//!
//! Patterns are represented as ASCII strings of `.` (dit) and `-` (dah).

/// The full set of supported (char, pattern) entries.
///
/// Covers letters A–Z, digits 0–9, and the most common punctuation.
/// Patterns use `'.'` for dit and `'-'` for dah.
pub const TABLE: &[(char, &str)] = &[
    ('A', ".-"),
    ('B', "-..."),
    ('C', "-.-."),
    ('D', "-.."),
    ('E', "."),
    ('F', "..-."),
    ('G', "--."),
    ('H', "...."),
    ('I', ".."),
    ('J', ".---"),
    ('K', "-.-"),
    ('L', ".-.."),
    ('M', "--"),
    ('N', "-."),
    ('O', "---"),
    ('P', ".--."),
    ('Q', "--.-"),
    ('R', ".-."),
    ('S', "..."),
    ('T', "-"),
    ('U', "..-"),
    ('V', "...-"),
    ('W', ".--"),
    ('X', "-..-"),
    ('Y', "-.--"),
    ('Z', "--.."),
    ('0', "-----"),
    ('1', ".----"),
    ('2', "..---"),
    ('3', "...--"),
    ('4', "....-"),
    ('5', "....."),
    ('6', "-...."),
    ('7', "--..."),
    ('8', "---.."),
    ('9', "----."),
    ('.', ".-.-.-"),
    (',', "--..--"),
    ('?', "..--.."),
    ('\'', ".----."),
    ('!', "-.-.--"),
    ('/', "-..-."),
    ('(', "-.--."),
    (')', "-.--.-"),
    ('&', ".-..."),
    (':', "---..."),
    (';', "-.-.-."),
    ('=', "-...-"),
    ('+', ".-.-."),
    ('-', "-....-"),
    ('_', "..--.-"),
    ('"', ".-..-."),
    ('$', "...-..-"),
    ('@', ".--.-."),
];

/// Look up the character for a pattern. Returns `None` for unknown patterns.
#[must_use]
pub fn char_for_pattern(pattern: &str) -> Option<char> {
    TABLE
        .iter()
        .find_map(|(ch, pat)| (*pat == pattern).then_some(*ch))
}

/// Look up the pattern for a character. Input is case-insensitive for letters.
#[must_use]
pub fn pattern_for_char(ch: char) -> Option<&'static str> {
    let upper = ch.to_ascii_uppercase();
    TABLE
        .iter()
        .find_map(|(c, pat)| (*c == upper).then_some(*pat))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_letters_roundtrip() {
        for ch in ['E', 'T', 'A', 'N', 'Q', 'Z'] {
            let pat = pattern_for_char(ch).unwrap();
            assert_eq!(char_for_pattern(pat), Some(ch));
        }
    }

    #[test]
    fn digits_roundtrip() {
        for ch in '0'..='9' {
            let pat = pattern_for_char(ch).unwrap();
            assert_eq!(char_for_pattern(pat), Some(ch));
        }
    }

    #[test]
    fn lowercase_input_is_accepted() {
        assert_eq!(pattern_for_char('q'), pattern_for_char('Q'));
    }

    #[test]
    fn unknown_pattern_returns_none() {
        assert_eq!(char_for_pattern("........"), None);
        assert_eq!(char_for_pattern(""), None);
    }

    #[test]
    fn unknown_char_returns_none() {
        assert_eq!(pattern_for_char('\u{00e9}'), None); // é
    }

    #[test]
    fn patterns_are_unique() {
        for (i, (_, a)) in TABLE.iter().enumerate() {
            for (_, b) in &TABLE[i + 1..] {
                assert_ne!(a, b, "duplicate pattern {a}");
            }
        }
    }
}
