// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Biome classification from climate data, shared by the F3 diagnostics
//! overlay, `/locate biome`, and `/tectonic`. This is a coarse, host-side
//! label for display only -- it does not drive block placement (that lives
//! in `cubic-game`, which will need its own copy when the biome refinement
//! card adds surface-block/vegetation variation).

/// Classify a biome from temperature, moisture, and elevation. Elevation
/// takes priority (alpine regardless of climate), then temperature bands,
/// then moisture within each band.
pub(crate) fn classify_biome(temp_c: f32, moisture_pct: f32, elevation_m: f32) -> &'static str {
    if elevation_m > 3500.0 {
        return "Alpine";
    }
    if temp_c < -8.0 {
        return "Glacier/Ice Sheet";
    }
    if temp_c < 0.0 {
        return if moisture_pct > 50.0 {
            "Tundra"
        } else {
            "Polar Desert"
        };
    }
    if temp_c < 10.0 {
        return if moisture_pct > 60.0 {
            "Boreal Forest"
        } else {
            "Cold Steppe"
        };
    }
    if temp_c < 20.0 {
        return if moisture_pct > 70.0 {
            "Temperate Rainforest"
        } else if moisture_pct > 40.0 {
            "Temperate Grassland"
        } else {
            "Shrubland"
        };
    }
    if moisture_pct > 75.0 {
        "Tropical Rainforest"
    } else if moisture_pct > 50.0 {
        "Tropical Forest"
    } else if moisture_pct > 25.0 {
        "Savanna"
    } else {
        "Desert"
    }
}

/// Label a tectonic boundary type value (-1.0 divergent .. +1.0 convergent).
pub(crate) fn classify_boundary(boundary_type: f32) -> &'static str {
    if boundary_type > 0.3 {
        "convergent"
    } else if boundary_type < -0.3 {
        "divergent"
    } else {
        "transform"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_elevation_is_always_alpine() {
        assert_eq!(classify_biome(25.0, 80.0, 4000.0), "Alpine");
    }

    #[test]
    fn very_cold_is_glacier() {
        assert_eq!(classify_biome(-10.0, 50.0, 100.0), "Glacier/Ice Sheet");
    }

    #[test]
    fn cold_and_dry_is_polar_desert() {
        assert_eq!(classify_biome(-4.0, 20.0, 100.0), "Polar Desert");
    }

    #[test]
    fn cold_and_wet_is_tundra() {
        assert_eq!(classify_biome(-4.0, 70.0, 100.0), "Tundra");
    }

    #[test]
    fn hot_and_dry_is_desert() {
        assert_eq!(classify_biome(30.0, 10.0, 50.0), "Desert");
    }

    #[test]
    fn hot_and_wet_is_tropical_rainforest() {
        assert_eq!(classify_biome(28.0, 85.0, 50.0), "Tropical Rainforest");
    }

    #[test]
    fn temperate_moderate_is_grassland() {
        assert_eq!(classify_biome(15.0, 50.0, 50.0), "Temperate Grassland");
    }

    #[test]
    fn boundary_classification_thresholds() {
        assert_eq!(classify_boundary(0.62), "convergent");
        assert_eq!(classify_boundary(-0.62), "divergent");
        assert_eq!(classify_boundary(0.0), "transform");
        assert_eq!(classify_boundary(0.3), "transform");
        assert_eq!(classify_boundary(0.31), "convergent");
    }
}
