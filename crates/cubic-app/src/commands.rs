// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Command dispatcher. Host-side built-ins + WASM game command delegation.

use crate::App;

pub(crate) fn dispatch(app: &mut App, input: &str) {
    // Strip leading slash
    let input = input.trim_start_matches('/');
    let mut parts = input.split_whitespace();
    let Some(cmd) = parts.next() else { return };
    let args: Vec<&str> = parts.collect();

    let result = match cmd {
        "tp" => cmd_tp(app, &args),
        "help" => cmd_help(&args),
        "locate" => Ok("Biome location not yet implemented.".to_string()),
        other => Err(format!("Unknown command: /{other}. Type /help for a list.")),
    };

    match result {
        Ok(msg) if !msg.is_empty() => {
            app.push_chat_message(msg, crate::ui::ChatMessageKind::CommandOutput);
        }
        Err(msg) => {
            app.push_chat_message(msg, crate::ui::ChatMessageKind::Error);
        }
        _ => {}
    }
}

fn cmd_tp(app: &mut App, args: &[&str]) -> Result<String, String> {
    // /tp x y z  or  /tp @p x y z  or  /tp @c x y z
    let (selector, coords) = if args.first().map(|a| a.starts_with('@')).unwrap_or(false) {
        (args[0], &args[1..])
    } else {
        ("@p", args)
    };

    if coords.len() != 3 {
        return Err("Usage: /tp [@p|@c] <x> <y> <z>".to_string());
    }

    let x = coords[0]
        .parse::<f64>()
        .map_err(|_| format!("Expected a number, got '{}'", coords[0]))?;
    let y = coords[1]
        .parse::<f64>()
        .map_err(|_| format!("Expected a number, got '{}'", coords[1]))?;
    let z = coords[2]
        .parse::<f64>()
        .map_err(|_| format!("Expected a number, got '{}'", coords[2]))?;

    match selector {
        "@p" => {
            cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                name: "teleport".to_string(),
                kind: 0,
                payload: [x, y, z],
            });
            Ok(format!("Teleported to {x:.1} {y:.1} {z:.1}"))
        }
        "@c" => {
            cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                name: "spectate".to_string(),
                kind: 0,
                payload: [0.0; 3],
            });
            app.camera.position = cubic_math::DVec3::new(x, y, z);
            Ok(format!("Camera moved to {x:.1} {y:.1} {z:.1}"))
        }
        s => Err(format!("Unknown selector '{s}'. Use @p or @c.")),
    }
}

fn cmd_help(args: &[&str]) -> Result<String, String> {
    if args.is_empty() {
        Ok("/tp [@p|@c] <x> <y> <z> — teleport\n/locate biome <name> — find biome\n/help [command] — show help".to_string())
    } else {
        match args[0] {
            "tp" => Ok("/tp [@p|@c] <x> <y> <z> — teleport player or camera".to_string()),
            "locate" => {
                Ok("/locate biome <name> — find nearest biome (not yet implemented)".to_string())
            }
            "help" => Ok("/help [command] — list commands or show usage for one".to_string()),
            other => Err(format!("Unknown command: /{other}")),
        }
    }
}
