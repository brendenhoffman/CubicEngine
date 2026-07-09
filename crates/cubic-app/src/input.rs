// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Physical input: keycode/mouse/gamepad string round-tripping, resolved
//! bindings, and the discrete-event tracker (tap/double-tap).

use crate::config::{AppCfg, CustomControl, KeyBinding, ModifierKey, TriggerKind};
use cubic_platform::winit::{event::MouseButton, keyboard::KeyCode};
use std::collections::HashSet;

// Note: pause (Escape) is intentionally not bindable here — it's hardcoded
// engine behavior for the app state machine, not a remappable control.
fn str_to_keycode(s: &str) -> Option<KeyCode> {
    match s {
        "KeyA" => Some(KeyCode::KeyA),
        "KeyB" => Some(KeyCode::KeyB),
        "KeyC" => Some(KeyCode::KeyC),
        "KeyD" => Some(KeyCode::KeyD),
        "KeyE" => Some(KeyCode::KeyE),
        "KeyF" => Some(KeyCode::KeyF),
        "KeyG" => Some(KeyCode::KeyG),
        "KeyH" => Some(KeyCode::KeyH),
        "KeyI" => Some(KeyCode::KeyI),
        "KeyJ" => Some(KeyCode::KeyJ),
        "KeyK" => Some(KeyCode::KeyK),
        "KeyL" => Some(KeyCode::KeyL),
        "KeyM" => Some(KeyCode::KeyM),
        "KeyN" => Some(KeyCode::KeyN),
        "KeyO" => Some(KeyCode::KeyO),
        "KeyP" => Some(KeyCode::KeyP),
        "KeyQ" => Some(KeyCode::KeyQ),
        "KeyR" => Some(KeyCode::KeyR),
        "KeyS" => Some(KeyCode::KeyS),
        "KeyT" => Some(KeyCode::KeyT),
        "KeyU" => Some(KeyCode::KeyU),
        "KeyV" => Some(KeyCode::KeyV),
        "KeyW" => Some(KeyCode::KeyW),
        "KeyX" => Some(KeyCode::KeyX),
        "KeyY" => Some(KeyCode::KeyY),
        "KeyZ" => Some(KeyCode::KeyZ),
        "Digit0" => Some(KeyCode::Digit0),
        "Digit1" => Some(KeyCode::Digit1),
        "Digit2" => Some(KeyCode::Digit2),
        "Digit3" => Some(KeyCode::Digit3),
        "Digit4" => Some(KeyCode::Digit4),
        "Digit5" => Some(KeyCode::Digit5),
        "Digit6" => Some(KeyCode::Digit6),
        "Digit7" => Some(KeyCode::Digit7),
        "Digit8" => Some(KeyCode::Digit8),
        "Digit9" => Some(KeyCode::Digit9),
        "Space" => Some(KeyCode::Space),
        "ShiftLeft" => Some(KeyCode::ShiftLeft),
        "ShiftRight" => Some(KeyCode::ShiftRight),
        "ControlLeft" => Some(KeyCode::ControlLeft),
        "ControlRight" => Some(KeyCode::ControlRight),
        "AltLeft" => Some(KeyCode::AltLeft),
        "AltRight" => Some(KeyCode::AltRight),
        "CapsLock" => Some(KeyCode::CapsLock),
        "Insert" => Some(KeyCode::Insert),
        "F1" => Some(KeyCode::F1),
        "F2" => Some(KeyCode::F2),
        "F3" => Some(KeyCode::F3),
        "F4" => Some(KeyCode::F4),
        "F5" => Some(KeyCode::F5),
        "F6" => Some(KeyCode::F6),
        "F7" => Some(KeyCode::F7),
        "F8" => Some(KeyCode::F8),
        "F9" => Some(KeyCode::F9),
        "F10" => Some(KeyCode::F10),
        "F11" => Some(KeyCode::F11),
        "F12" => Some(KeyCode::F12),
        "ArrowUp" => Some(KeyCode::ArrowUp),
        "ArrowDown" => Some(KeyCode::ArrowDown),
        "ArrowLeft" => Some(KeyCode::ArrowLeft),
        "ArrowRight" => Some(KeyCode::ArrowRight),
        // winit's variant is `Enter`, not `Return`.
        "Enter" => Some(KeyCode::Enter),
        "Tab" => Some(KeyCode::Tab),
        "Backspace" => Some(KeyCode::Backspace),
        "Delete" => Some(KeyCode::Delete),
        "Home" => Some(KeyCode::Home),
        "End" => Some(KeyCode::End),
        "PageUp" => Some(KeyCode::PageUp),
        "PageDown" => Some(KeyCode::PageDown),
        _ => {
            tracing::warn!("unknown key name in config: {s}");
            None
        }
    }
}

/// Reverse of `str_to_keycode` — used by the Controls tab's remap capture
/// (see `WindowEvent::KeyboardInput`) to turn a captured `KeyCode` back into
/// the string stored in config. Deliberately exhaustive over the same set
/// `str_to_keycode` accepts, including modifiers (ShiftLeft/ControlLeft/
/// AltLeft/etc): unlike egui's `Key` enum, winit's `KeyCode` reports bare
/// modifier presses as ordinary keys with a precise left/right physical
/// side, which is what makes capturing them for a binding possible at all.
pub(crate) fn keycode_to_str(code: KeyCode) -> Option<&'static str> {
    Some(match code {
        KeyCode::KeyA => "KeyA",
        KeyCode::KeyB => "KeyB",
        KeyCode::KeyC => "KeyC",
        KeyCode::KeyD => "KeyD",
        KeyCode::KeyE => "KeyE",
        KeyCode::KeyF => "KeyF",
        KeyCode::KeyG => "KeyG",
        KeyCode::KeyH => "KeyH",
        KeyCode::KeyI => "KeyI",
        KeyCode::KeyJ => "KeyJ",
        KeyCode::KeyK => "KeyK",
        KeyCode::KeyL => "KeyL",
        KeyCode::KeyM => "KeyM",
        KeyCode::KeyN => "KeyN",
        KeyCode::KeyO => "KeyO",
        KeyCode::KeyP => "KeyP",
        KeyCode::KeyQ => "KeyQ",
        KeyCode::KeyR => "KeyR",
        KeyCode::KeyS => "KeyS",
        KeyCode::KeyT => "KeyT",
        KeyCode::KeyU => "KeyU",
        KeyCode::KeyV => "KeyV",
        KeyCode::KeyW => "KeyW",
        KeyCode::KeyX => "KeyX",
        KeyCode::KeyY => "KeyY",
        KeyCode::KeyZ => "KeyZ",
        KeyCode::Digit0 => "Digit0",
        KeyCode::Digit1 => "Digit1",
        KeyCode::Digit2 => "Digit2",
        KeyCode::Digit3 => "Digit3",
        KeyCode::Digit4 => "Digit4",
        KeyCode::Digit5 => "Digit5",
        KeyCode::Digit6 => "Digit6",
        KeyCode::Digit7 => "Digit7",
        KeyCode::Digit8 => "Digit8",
        KeyCode::Digit9 => "Digit9",
        KeyCode::Space => "Space",
        KeyCode::ShiftLeft => "ShiftLeft",
        KeyCode::ShiftRight => "ShiftRight",
        KeyCode::ControlLeft => "ControlLeft",
        KeyCode::ControlRight => "ControlRight",
        KeyCode::AltLeft => "AltLeft",
        KeyCode::AltRight => "AltRight",
        KeyCode::CapsLock => "CapsLock",
        KeyCode::Insert => "Insert",
        KeyCode::F1 => "F1",
        KeyCode::F2 => "F2",
        KeyCode::F3 => "F3",
        KeyCode::F4 => "F4",
        KeyCode::F5 => "F5",
        KeyCode::F6 => "F6",
        KeyCode::F7 => "F7",
        KeyCode::F8 => "F8",
        KeyCode::F9 => "F9",
        KeyCode::F10 => "F10",
        KeyCode::F11 => "F11",
        KeyCode::F12 => "F12",
        KeyCode::ArrowUp => "ArrowUp",
        KeyCode::ArrowDown => "ArrowDown",
        KeyCode::ArrowLeft => "ArrowLeft",
        KeyCode::ArrowRight => "ArrowRight",
        KeyCode::Enter => "Enter",
        KeyCode::Tab => "Tab",
        KeyCode::Backspace => "Backspace",
        KeyCode::Delete => "Delete",
        KeyCode::Home => "Home",
        KeyCode::End => "End",
        KeyCode::PageUp => "PageUp",
        KeyCode::PageDown => "PageDown",
        _ => return None,
    })
}

/// Any physical input that can be captured as a control's base binding —
/// keyboard key, mouse button, or gamepad button. A binding's modifier
/// (ModifierKey::Shift/Control/Alt) is always keyboard-only regardless of
/// which source this is, same as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum InputSource {
    Key(KeyCode),
    Mouse(MouseButton),
    Gamepad(gilrs::Button),
}

fn mouse_button_to_string(button: MouseButton) -> String {
    match button {
        MouseButton::Left => "MouseLeft".to_string(),
        MouseButton::Right => "MouseRight".to_string(),
        MouseButton::Middle => "MouseMiddle".to_string(),
        MouseButton::Back => "MouseBack".to_string(),
        MouseButton::Forward => "MouseForward".to_string(),
        MouseButton::Other(n) => format!("MouseOther{n}"),
    }
}

fn str_to_mouse_button(s: &str) -> Option<MouseButton> {
    Some(match s {
        "MouseLeft" => MouseButton::Left,
        "MouseRight" => MouseButton::Right,
        "MouseMiddle" => MouseButton::Middle,
        "MouseBack" => MouseButton::Back,
        "MouseForward" => MouseButton::Forward,
        other => {
            let n: u16 = other.strip_prefix("MouseOther")?.parse().ok()?;
            MouseButton::Other(n)
        }
    })
}

/// gilrs::Button::Unknown deliberately has no string form — a binding can't
/// meaningfully be restored to "some unrecognized button", so it's simply
/// not capturable/round-trippable, same as an unrecognized KeyCode.
fn gamepad_button_to_str(button: gilrs::Button) -> Option<&'static str> {
    use gilrs::Button::*;
    Some(match button {
        South => "GamepadSouth",
        East => "GamepadEast",
        North => "GamepadNorth",
        West => "GamepadWest",
        C => "GamepadC",
        Z => "GamepadZ",
        LeftTrigger => "GamepadLeftTrigger",
        LeftTrigger2 => "GamepadLeftTrigger2",
        RightTrigger => "GamepadRightTrigger",
        RightTrigger2 => "GamepadRightTrigger2",
        Select => "GamepadSelect",
        Start => "GamepadStart",
        Mode => "GamepadMode",
        LeftThumb => "GamepadLeftThumb",
        RightThumb => "GamepadRightThumb",
        DPadUp => "GamepadDPadUp",
        DPadDown => "GamepadDPadDown",
        DPadLeft => "GamepadDPadLeft",
        DPadRight => "GamepadDPadRight",
        Unknown => return None,
    })
}

fn str_to_gamepad_button(s: &str) -> Option<gilrs::Button> {
    use gilrs::Button::*;
    Some(match s {
        "GamepadSouth" => South,
        "GamepadEast" => East,
        "GamepadNorth" => North,
        "GamepadWest" => West,
        "GamepadC" => C,
        "GamepadZ" => Z,
        "GamepadLeftTrigger" => LeftTrigger,
        "GamepadLeftTrigger2" => LeftTrigger2,
        "GamepadRightTrigger" => RightTrigger,
        "GamepadRightTrigger2" => RightTrigger2,
        "GamepadSelect" => Select,
        "GamepadStart" => Start,
        "GamepadMode" => Mode,
        "GamepadLeftThumb" => LeftThumb,
        "GamepadRightThumb" => RightThumb,
        "GamepadDPadUp" => DPadUp,
        "GamepadDPadDown" => DPadDown,
        "GamepadDPadLeft" => DPadLeft,
        "GamepadDPadRight" => DPadRight,
        _ => return None,
    })
}

/// Config-string form of an `InputSource`, for writing a freshly captured
/// binding back into cubic.toml/profile.toml (see `apply_control_remap`).
pub(crate) fn input_source_to_string(source: InputSource) -> Option<String> {
    match source {
        InputSource::Key(code) => keycode_to_str(code).map(str::to_string),
        InputSource::Mouse(button) => Some(mouse_button_to_string(button)),
        InputSource::Gamepad(button) => gamepad_button_to_str(button).map(str::to_string),
    }
}

/// Reverse of `input_source_to_string`, tried in Mouse/Gamepad/Key order
/// since those prefixes are mutually exclusive with keyboard key names.
pub(crate) fn str_to_input_source(s: &str) -> Option<InputSource> {
    if s.starts_with("Mouse") {
        return str_to_mouse_button(s).map(InputSource::Mouse);
    }
    if s.starts_with("Gamepad") {
        return str_to_gamepad_button(s).map(InputSource::Gamepad);
    }
    str_to_keycode(s).map(InputSource::Key)
}

/// A control's binding resolved once at startup (see `resolve_controls`), so
/// the hot path doesn't re-parse strings every frame. `source: None` means
/// unbound — the control simply never activates, rather than silently
/// falling back to some default key (which would hide a broken/cleared
/// binding instead of surfacing it).
#[derive(Copy, Clone)]
pub(crate) struct ResolvedBinding {
    pub(crate) source: Option<InputSource>,
    pub(crate) modifier: ModifierKey,
    pub(crate) trigger: TriggerKind,
}

fn resolve_binding(b: &KeyBinding) -> ResolvedBinding {
    ResolvedBinding {
        source: b.key.as_deref().and_then(str_to_input_source),
        modifier: b.modifier,
        trigger: b.trigger,
    }
}

#[derive(Copy, Clone)]
pub(crate) struct ResolvedControls {
    pub(crate) forward: ResolvedBinding,
    pub(crate) back: ResolvedBinding,
    pub(crate) left: ResolvedBinding,
    pub(crate) right: ResolvedBinding,
    pub(crate) jump: ResolvedBinding,
    pub(crate) sneak: ResolvedBinding,
    pub(crate) toggle_diagnostics: ResolvedBinding,
    pub(crate) toggle_third_person: ResolvedBinding,
    pub(crate) spectate: ResolvedBinding,
    pub(crate) fly: ResolvedBinding,
}

pub(crate) fn resolve_controls(cfg: &AppCfg) -> ResolvedControls {
    ResolvedControls {
        forward: resolve_binding(&cfg.controls.forward),
        back: resolve_binding(&cfg.controls.back),
        left: resolve_binding(&cfg.controls.left),
        right: resolve_binding(&cfg.controls.right),
        jump: resolve_binding(&cfg.controls.jump),
        sneak: resolve_binding(&cfg.controls.sneak),
        toggle_diagnostics: resolve_binding(&cfg.controls.toggle_diagnostics),
        toggle_third_person: resolve_binding(&cfg.controls.toggle_third_person),
        spectate: resolve_binding(&cfg.controls.spectate),
        fly: resolve_binding(&cfg.controls.fly),
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

pub(crate) const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

#[derive(Default)]
pub(crate) struct InputState {
    held: HashSet<InputSource>,
    // Sources that saw at least one Pressed transition since the last time
    // InputTracker::update drained this (see binding_pressed_this_tick).
    // Exists because InputTracker only samples state once per rendered
    // frame — a source that's pressed *and* released again within that
    // same frame would otherwise read as "never held" and the tap would
    // silently vanish. This isn't just theoretical: some trackpoint
    // drivers (e.g. ThinkPads with click-to-scroll on the middle button)
    // synthesize a very short press/release pair around what the user
    // experiences as a single deliberate click, well under one frame long.
    pressed_since_check: HashSet<InputSource>,
    mouse_delta: (f32, f32),
}

impl InputState {
    pub(crate) fn set_source(&mut self, source: InputSource, pressed: bool) {
        if pressed {
            self.held.insert(source);
            self.pressed_since_check.insert(source);
        } else {
            self.held.remove(&source);
        }
    }

    pub(crate) fn is_held(&self, source: InputSource) -> bool {
        self.held.contains(&source)
    }

    /// Whether the given modifier is currently held, side-agnostic (either
    /// ShiftLeft or ShiftRight counts as "Shift"). `ModifierKey::None`
    /// trivially always holds — a binding with no modifier configured
    /// shouldn't require one. Modifiers are always keyboard keys regardless
    /// of what source the binding's own base key uses.
    pub(crate) fn modifier_held(&self, modifier: ModifierKey) -> bool {
        match modifier {
            ModifierKey::None => true,
            ModifierKey::Shift => {
                self.is_held(InputSource::Key(KeyCode::ShiftLeft))
                    || self.is_held(InputSource::Key(KeyCode::ShiftRight))
            }
            ModifierKey::Control => {
                self.is_held(InputSource::Key(KeyCode::ControlLeft))
                    || self.is_held(InputSource::Key(KeyCode::ControlRight))
            }
            ModifierKey::Alt => {
                self.is_held(InputSource::Key(KeyCode::AltLeft))
                    || self.is_held(InputSource::Key(KeyCode::AltRight))
            }
        }
    }

    /// Whether a resolved binding is currently "held": its base input
    /// source is down (if any — unbound bindings never activate) and its
    /// configured modifier, if any, is also down.
    pub(crate) fn binding_active(&self, binding: &ResolvedBinding) -> bool {
        match binding.source {
            Some(source) => self.is_held(source) && self.modifier_held(binding.modifier),
            None => false,
        }
    }

    /// Like `binding_active`, but also counts as "held" if the source was
    /// pressed at any point since the last `clear_pressed_since_check` call
    /// even if it's already been released again — see `pressed_since_check`'s
    /// doc comment for why InputTracker needs this instead of the plain
    /// instant-in-time check movement controls use.
    fn binding_pressed_this_tick(&self, binding: &ResolvedBinding) -> bool {
        match binding.source {
            Some(source) => {
                (self.is_held(source) || self.pressed_since_check.contains(&source))
                    && self.modifier_held(binding.modifier)
            }
            None => false,
        }
    }

    /// Drain `pressed_since_check` — called once per frame by
    /// `InputTracker::update` after it's done consulting
    /// `binding_pressed_this_tick`, so a same-frame press+release doesn't
    /// keep reading as "held" forever afterward.
    fn clear_pressed_since_check(&mut self) {
        self.pressed_since_check.clear();
    }

    pub(crate) fn accumulate_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Returns the accumulated delta and resets it to zero.
    pub(crate) fn take_mouse_delta(&mut self) -> (f32, f32) {
        std::mem::take(&mut self.mouse_delta)
    }

    /// Clears every held source — used when key-up events can no longer be
    /// reliably observed (e.g. window unfocused), so movement doesn't get
    /// stuck on alt-tab.
    pub(crate) fn clear_held(&mut self) {
        self.held.clear();
    }
}

pub(crate) struct ActionTracker {
    pub(crate) was_held: bool,
    pub(crate) last_press_time: f32,
}

/// Tracks the purely discrete/toggle-style controls only (movement's
/// forward/back/left/right/jump/sneak are read continuously via
/// `InputState::binding_active` directly into `InputSnapshot` instead —
/// they have no use for tap/double-tap/hold gating, so pushing events for
/// them would just be wasted WASM-boundary traffic).
pub(crate) struct InputTracker {
    // (action_name, binding, state)
    pub(crate) actions: Vec<(String, ResolvedBinding, ActionTracker)>,
    pub(crate) elapsed: f32,
}

impl InputTracker {
    /// `custom` are controls the currently loaded game registered itself
    /// (see CustomControl) — tracked exactly like the four built-in
    /// discrete controls, just resolved from their own KeyBinding instead
    /// of a ResolvedControls field, since the set of names isn't known at
    /// compile time.
    pub(crate) fn new(controls: &ResolvedControls, custom: &[CustomControl]) -> Self {
        let mut actions = vec![
            (
                "toggle_diagnostics".into(),
                controls.toggle_diagnostics,
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ),
            (
                "toggle_third_person".into(),
                controls.toggle_third_person,
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ),
            (
                "spectate".into(),
                controls.spectate,
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ),
            (
                "fly".into(),
                controls.fly,
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ),
        ];
        for c in custom {
            actions.push((
                c.name.clone(),
                resolve_binding(&c.binding),
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ));
        }
        Self {
            actions,
            elapsed: 0.0,
        }
    }

    /// Advances edge/double-tap detection for every tracked action and
    /// returns the names of any whose configured trigger condition (see
    /// TriggerKind) was satisfied this tick — a lone tap on a
    /// DoubleTap-configured control is *not* included, only a genuine rapid
    /// double-press is. Every fired action is also forwarded to the guest
    /// as an InputEvent (kind 0, or 2 if it happened to be a double-tap);
    /// callers that only care about something host-side (toggle_diagnostics)
    /// use the returned names instead of waiting on a guest round-trip.
    pub(crate) fn update(&mut self, input: &mut InputState, dt: f32) -> Vec<String> {
        self.elapsed += dt;
        let mut fired = Vec::new();
        for (name, binding, state) in &mut self.actions {
            let is_held = input.binding_pressed_this_tick(binding);
            if is_held && !state.was_held {
                let is_double_tap = self.elapsed - state.last_press_time < 0.3;
                state.last_press_time = self.elapsed;
                // DoubleTap is the one case that actually changes *whether*
                // the action fires, not just which kind is reported: a lone
                // tap is swallowed here rather than forwarded, so a
                // DoubleTap-configured control (unlike Hold/Tap) never
                // activates on a single press.
                let should_fire = match binding.trigger {
                    TriggerKind::DoubleTap => is_double_tap,
                    TriggerKind::Tap => true,
                };
                if should_fire {
                    cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                        name: name.clone(),
                        kind: if is_double_tap { 2 } else { 0 },
                    });
                    fired.push(name.clone());
                }
            } else if !is_held && state.was_held {
                cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                    name: name.clone(),
                    kind: 1, // Released
                });
            }
            state.was_held = is_held;
        }
        input.clear_pressed_since_check();
        fired
    }
}
