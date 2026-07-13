// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Command dispatcher. Host-side built-ins + WASM game command delegation.

use crate::ui::ChatMessageKind;
use crate::App;

pub(crate) fn dispatch(app: &mut App, input: &str) {
    let input = input.trim_start_matches('/').trim();
    let mut parts = input.split_whitespace();
    let Some(cmd) = parts.next() else { return };
    let args: Vec<&str> = parts.collect();

    let result = match cmd {
        "tp" => cmd_tp(app, &args),
        "set" => cmd_set(app, &args),
        "help" => cmd_help(app, &args),
        "locate" => Ok("Biome location not yet implemented.".to_string()),
        other => {
            // Check game-registered commands
            if let Some(cmd) = app
                .guest
                .registered_commands
                .iter()
                .find(|c| c.name == other)
            {
                let id = cmd.command_id;
                let result = app
                    .guest
                    .wasm_game
                    .as_ref()
                    .map(|g| g.on_command(id, args.iter().map(|s| s.to_string()).collect()))
                    .unwrap_or_default();
                if !result.is_empty() {
                    app.push_chat_message(result, ChatMessageKind::CommandOutput);
                }
                return;
            }
            Err(format!("Unknown command: /{other}. Type /help for a list."))
        }
    };

    match result {
        Ok(msg) if !msg.is_empty() => {
            app.push_chat_message(msg, ChatMessageKind::CommandOutput);
        }
        Err(msg) => {
            app.push_chat_message(msg, ChatMessageKind::Error);
        }
        _ => {}
    }
}

/// Returns completion candidates for the current token at `cursor_pos`.
/// Called by the input bar on Tab and while the completion popup is open.
pub(crate) fn completions(app: &App, input: &str, cursor_pos: usize) -> Vec<String> {
    let before_cursor = &input[..cursor_pos.min(input.len())];
    let trimmed = before_cursor.trim_start_matches('/');
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let ends_with_space = before_cursor.ends_with(' ');

    // Completing the command name itself
    if tokens.is_empty() || (tokens.len() == 1 && !ends_with_space) {
        let partial = tokens.first().copied().unwrap_or("");
        let mut matches: Vec<String> = ["tp", "set", "help", "locate"]
            .iter()
            .filter(|c| c.starts_with(partial))
            .map(|c| format!("/{c}"))
            .collect();
        // Add game-registered commands
        for cmd in &app.guest.registered_commands {
            if cmd.name.starts_with(partial) {
                matches.push(format!("/{}", cmd.name));
            }
        }
        return matches;
    }

    let cmd = tokens[0];
    let arg_index = if ends_with_space {
        tokens.len() - 1
    } else {
        tokens.len() - 2
    };
    let partial = if ends_with_space {
        ""
    } else {
        tokens.last().copied().unwrap_or("")
    };

    match cmd {
        "tp" => {
            if arg_index == 0 {
                // First arg: selector or coordinate
                vec!["@p", "@c", "~"]
                    .into_iter()
                    .filter(|s| s.starts_with(partial))
                    .map(String::from)
                    .collect()
            } else {
                // Coordinate positions
                vec!["~".to_string()]
                    .into_iter()
                    .filter(|s| s.starts_with(partial))
                    .collect()
            }
        }
        "set" => {
            if arg_index == 0 {
                let keys = [
                    "fly_speed",
                    "walk_speed",
                    "jump_velocity",
                    "gravity",
                    "sprint_multiplier",
                    "mouse_sensitivity",
                ];
                keys.iter()
                    .filter(|k| k.starts_with(partial))
                    .map(|k| k.to_string())
                    .collect()
            } else {
                vec![]
            }
        }
        "help" => {
            let builtins = ["tp", "set", "help", "locate"];
            builtins
                .iter()
                .filter(|c| c.starts_with(partial))
                .map(|c| c.to_string())
                .collect()
        }
        _ => {
            // Game-registered command completions
            if let Some(cmd) = app.guest.registered_commands.iter().find(|c| c.name == cmd) {
                if let Some(values) = cmd.completions.get(&(arg_index as u32)) {
                    return values
                        .iter()
                        .filter(|v| v.starts_with(partial))
                        .cloned()
                        .collect();
                }
            }
            vec![]
        }
    }
}

// ---------------------------------------------------------------------------
// /tp
// ---------------------------------------------------------------------------

fn cmd_tp(app: &mut App, args: &[&str]) -> Result<String, String> {
    // Determine selector and coordinate tokens
    let (selector, coord_tokens) = if args.first().map(|a| a.starts_with('@')).unwrap_or(false) {
        (args[0], &args[1..])
    } else {
        // Smart default: @c in spectator, @p otherwise
        let default = if app.player_spectating { "@c" } else { "@p" };
        (default, args)
    };

    if coord_tokens.len() != 3 {
        return Err("Usage: /tp [@p|@c] <x> <y> <z>  (use ~ for relative coords)".to_string());
    }

    // Resolve current position for ~ expansion
    let current = match selector {
        "@p" => {
            let feet = cubic_wasm::get_player_feet();
            [feet.x, feet.y, feet.z]
        }
        "@c" => {
            let p = app.camera.position;
            [p.x, p.y, p.z]
        }
        s => return Err(format!("Unknown selector '{s}'. Use @p or @c.")),
    };

    let x = resolve_coord(coord_tokens[0], current[0])?;
    let y = resolve_coord(coord_tokens[1], current[1])?;
    let z = resolve_coord(coord_tokens[2], current[2])?;

    match selector {
        "@p" => {
            cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                name: "teleport".to_string(),
                kind: 0,
                payload: format!("{x} {y} {z}"),
            });
            Ok(format!("Teleported to {x:.1} {y:.1} {z:.1}"))
        }
        "@c" => {
            if !app.player_spectating {
                cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                    name: "spectate".to_string(),
                    kind: 0,
                    payload: format!("{x} {y} {z}"), // target position
                });
            } else {
                cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                    name: "teleport_spectator".to_string(),
                    kind: 0,
                    payload: format!("{x} {y} {z}"),
                });
            }
            Ok(format!("Camera moved to {x:.1} {y:.1} {z:.1}"))
        }
        _ => unreachable!(),
    }
}

fn resolve_coord(token: &str, current: f64) -> Result<f64, String> {
    if token == "~" {
        Ok(current)
    } else if let Some(rest) = token.strip_prefix('~') {
        rest.parse::<f64>()
            .map(|offset| current + offset)
            .map_err(|_| format!("Expected a number after ~, got '{rest}'"))
    } else {
        token
            .parse::<f64>()
            .map_err(|_| format!("Expected a number, got '{token}'"))
    }
}

// ---------------------------------------------------------------------------
// /set
// ---------------------------------------------------------------------------

fn cmd_set(app: &mut App, args: &[&str]) -> Result<String, String> {
    if args.is_empty() {
        return Ok(format!(
            "fly_speed={:.2}  walk_speed={:.2}  jump_velocity={:.2}  \
             gravity={:.2}  sprint_multiplier={:.2}  mouse_sensitivity={:.4}",
            app.cfg.player.fly_speed,
            app.cfg.player.walk_speed,
            app.cfg.player.jump_velocity,
            app.cfg.player.gravity,
            app.cfg.player.sprint_multiplier,
            app.cfg.camera.mouse_sensitivity,
        ));
    }

    if args.len() != 2 {
        return Err("Usage: /set <key> <value>  or  /set  (list current values)".to_string());
    }

    let key = args[0];
    let val: f32 = args[1]
        .parse()
        .map_err(|_| format!("Expected a number, got '{}'", args[1]))?;

    match key {
        "fly_speed" => app.cfg.player.fly_speed = val,
        "walk_speed" => app.cfg.player.walk_speed = val,
        "jump_velocity" => app.cfg.player.jump_velocity = val,
        "gravity" => app.cfg.player.gravity = val,
        "sprint_multiplier" => app.cfg.player.sprint_multiplier = val,
        "mouse_sensitivity" => app.cfg.camera.mouse_sensitivity = val,
        other => return Err(format!("Unknown setting '{other}'. Type /set for a list.")),
    }

    Ok(format!("{key} = {val}"))
}

// ---------------------------------------------------------------------------
// /help
// ---------------------------------------------------------------------------

fn cmd_help(app: &App, args: &[&str]) -> Result<String, String> {
    if args.is_empty() {
        let mut out = "/tp [@p|@c] <x> <y> <z> — teleport (~ for relative)\n\
              /set [<key> <value>] — view/change hot config\n\
              /locate biome <name> — find biome (not yet implemented)\n\
              /help [command] — show help"
            .to_string();
        if !app.guest.registered_commands.is_empty() {
            out.push_str("\nGame commands:");
            for cmd in &app.guest.registered_commands {
                out.push_str(&format!("\n  {} — {}", cmd.usage, cmd.description));
            }
        }
        Ok(out)
    } else {
        match args[0] {
            "tp" => Ok("/tp [@p|@c] <x> <y> <z> — teleport player or camera. \
                        Use ~ for relative coords, e.g. /tp ~ ~10 ~"
                .to_string()),
            "set" => Ok("/set — list hot config values\n\
                         /set <key> <value> — change for this session only\n\
                         Keys: fly_speed walk_speed jump_velocity gravity \
                         sprint_multiplier mouse_sensitivity"
                .to_string()),
            "locate" => {
                Ok("/locate biome <name> — find nearest biome (not yet implemented)".to_string())
            }
            "help" => Ok("/help [command] — list commands or show usage for one".to_string()),
            other => {
                if let Some(cmd) = app
                    .guest
                    .registered_commands
                    .iter()
                    .find(|c| c.name == other)
                {
                    return Ok(format!("{} — {}", cmd.usage, cmd.description));
                }
                Err(format!("Unknown command: /{other}"))
            }
        }
    }
}
