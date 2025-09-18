// SPDX-License-Identifier: CEPL-1.0

# Contributing to CubicEngine

We welcome contributions of code, assets, and mods!

> **Note on AI use:** AI generated assets will not be accepted into the Project. Use of AI for writing code will not be screened for and is recognized to be helpful for learning and troubleshooting. However, "Vibe Coding" is heavily frowned upon and any bad or faulty code will not be accepted into the project.

## Code Contributions

- Fork, branch, and submit a pull request.
- By submitting code that is merged, you agree to license your contribution under the Project's custom license: [CubicEngine Public License](LICENSE.md).
- Dependencies must use **OSI-approved** or **copyleft (GPL/AGPL/LGPL/MPL)** licenses.
  - **Non-OSS / source-available / non-commercial** licenses (e.g., SSPL, BUSL, Elastic, Prosperity) are **not permitted**.
  - Use `cargo-deny` to verify Rust specific dependencies.
- Code should be formatted to common standards, such as with tools like `rustfmt`, `stylua`, and `taplo`.
- Rust files should contain `#![deny(unsafe_op_in_unsafe_fn)]`.
  - Pull requests will be checked with `clippy`, but you can ease the process by running it yourself with the following targets: `aarch64-apple-darwin`, `aarch64-pc-windows-msvc`, `aarch64-unknown-linux-gnu`, `armv7-unknown-linux-gnueabihf`, `i686-unknown-linux-gnu`, `powerpc64le-unknown-linux-gnu`, `riscv64gc-unknown-linux-gnu`, `s390x-unknown-linux-gnu`, `x86_64-apple-darwin`, `x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`.
- Code with warnings will not be accepted.
- Once merged, your code becomes part of the Project and cannot be relicensed.
- By contributing, you grant an irrevocable, perpetual, non-exclusive, worldwide, royalty-free patent license under any patents you control that would otherwise be infringed by your contribution.
  This ensures the Project remains usable and safe from patent claims.
- This Project does not require or accept Contributor License Agreements (CLAs).
  All contributions are licensed directly under CEPL to prevent mixed licensing or proprietary carve-outs.

## Developer Certificate of Origin (DCO)

This project uses the Developer Certificate of Origin (DCO).
By contributing, you certify that:

- You wrote the code or have the right to submit it under the Project’s license.
- You are submitting it under the CubicEngine Public License (CEPL).
- You understand and agree that once merged, your contribution becomes part of the Project and cannot be relicensed separately.

All commits must include a "Signed-off-by" line (`git commit -s`).

## Asset Contributions

- See [Asset License & Contributor Agreement](ASSETS_LICENSE.md).
- Artists retain ownership but grant the Project a license to include the work in official releases.
- Assets may be withdrawn later at the contributor’s discretion.

## Modding

- Mods are independent of the core Project.
- Mods can use **any license** the author chooses, provided it does not conflict with the Project’s rules (e.g., must not include closed-source or monetized code).
- To be listed in the **official mod hub**, mods must:
  - Be open source.
  - Comply with the Project’s rules (no monetization, no intolerance, no harmful/illegal content).
- Modders retain full ownership and rights to their work.
- Mods adopted by the core Project (through merge or direct inclusion) are treated as contributions and re-licensed under the Project’s CubicEngine Public License.

### Adoption Consent

If a mod is submitted for inclusion in the core Project, the contributor must:

- Confirm they have the authority to relicense the mod under CEPL.
- Ensure all included code and assets are properly licensed or replaced.
- Acknowledge that once merged, the mod becomes part of the Project and cannot be relicensed separately.

---

Thank you for contributing to a welcoming, open, and donation-driven ecosystem!
