#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementState {
    Pressed,
    Released,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamedKey {
    Space,
    Shift,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    Character(String),
    Named(NamedKey),
    Other,
}
